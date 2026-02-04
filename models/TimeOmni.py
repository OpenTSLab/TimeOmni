from math import sqrt
import json

import torch
import torch.nn as nn
from torch import Tensor

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils.tools import dotdict

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        # Support both single pred_len and list of pred_lens
        if isinstance(configs.pred_len, (list, tuple)):
            self.pred_lens = list(configs.pred_len)
        else:
            self.pred_lens = [configs.pred_len]
        self.d_ff = configs.d_ff
        self.top_k = 5

        # analysis params (support list[int])
        apl = configs.analysis_patch_len
        ast = configs.analysis_stride
        if isinstance(apl, int):
            self.analysis_patch_lens = [apl]
        else:
            self.analysis_patch_lens = list(apl)
        if isinstance(ast, int):
            self.analysis_strides = [ast] * len(self.analysis_patch_lens)
        else:
            self.analysis_strides = list(ast)
            if len(self.analysis_strides) == 1 and len(self.analysis_patch_lens) > 1:
                self.analysis_strides = self.analysis_strides * len(self.analysis_patch_lens)
        if len(self.analysis_strides) != len(self.analysis_patch_lens):
            raise ValueError(f"analysis_stride (len={len(self.analysis_strides)}) must have same length as analysis_patch_len (len={len(self.analysis_patch_lens)}) or be a single value")

        # Load LLM model and tokenizer
        self._load_llm_model(configs)

        # Build multiple analysis patch embedders with a fixed d_model, and a single reprogramming layer
        self.analysis_d_model = getattr(configs, 'analysis_d_model', 512)
        self.analysis_patch_embeddings = nn.ModuleDict()
        for pl, st in zip(self.analysis_patch_lens, self.analysis_strides):
            self.analysis_patch_embeddings[str(pl)] = PatchEmbedding(
                self.analysis_d_model, pl, st, configs.dropout
            )
        # Check if MLP-based reprogramming is requested
        use_mlp = getattr(configs, 'use_mlp_reprogramming', False)
        self.analysis_reprogramming_layer = ReprogrammingLayer(
            self.analysis_d_model, configs.n_heads, None, self.d_llm, use_mlp=use_mlp
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.patch_nums = configs.patch_nums # 50
        self.ts_tokens = configs.ts_tokens # 100
        self.head_nf = self.d_ff * self.patch_nums

        # Create multiple output projections for different pred_lens
        self.output_projections = nn.ModuleDict()
        for pred_len in self.pred_lens:
            self.output_projections[str(pred_len)] = FlattenHead(self.head_nf, pred_len, head_dropout=configs.dropout)

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def _load_llm_model(self, configs):
        """
        Load and configure the LLM model and tokenizer based on configuration.
        
        Args:
            configs: Configuration object containing LLM settings
        """
        if configs.llm_model.lower() in ['qwen3', 'qwen-3', 'qwen3-8b']:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                './pretrained/Qwen3-8B',
                trust_remote_code=True,
                local_files_only=True,
                # output_attentions=True,
                output_hidden_states=True
            )
        
            self.tokenizer = AutoTokenizer.from_pretrained(
                './pretrained/Qwen3-8B',
                trust_remote_code=True,
                local_files_only=True
            )

        else:
            raise Exception(f'LLM model: {configs.llm_model} is not supported yet!')

        # Apply DoRA if requested
        if hasattr(configs, 'use_dora') and configs.use_dora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                
                # Parse target modules
                target_modules = configs.dora_target_modules.split(',')
                target_modules = [module.strip() for module in target_modules]
                
                # Create DoRA configuration
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=configs.dora_r,
                    lora_alpha=configs.dora_alpha,
                    lora_dropout=configs.dora_dropout,
                    target_modules=target_modules,
                    use_dora=True,  # Enable DoRA
                )
                
                # Apply PEFT to the model
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                print(f"Applied DoRA to LLM with r={configs.dora_r}, alpha={configs.dora_alpha}, target_modules={target_modules}")
                
                # Print trainable parameters
                self.llm_model.print_trainable_parameters()
                
            except ImportError:
                print("Warning: PEFT not available. Falling back to full parameter freezing.")
                for param in self.llm_model.parameters():
                    param.requires_grad = False
        elif hasattr(configs, 'full_finetune') and configs.full_finetune:
            # Full finetuning: all parameters are trainable
            pass
        else:
            # Original behavior: freeze all LLM parameters
            for param in self.llm_model.parameters():
                param.requires_grad = False

        # Get hidden size from model config
        if hasattr(self.llm_model, 'config'):
            self.d_llm = self.llm_model.config.hidden_size
        elif hasattr(self.llm_model, 'base_model') and hasattr(self.llm_model.base_model, 'config'):
            # For PEFT models
            self.d_llm = self.llm_model.base_model.config.hidden_size
        else:
            raise AttributeError("Cannot determine LLM hidden size")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Add time series special tokens
        self.TS_START_TOKEN = "<ts>"
        self.TS_END_TOKEN = "</ts>"
        special_tokens_dict = {'additional_special_tokens': [self.TS_START_TOKEN, self.TS_END_TOKEN]}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def forward(self, time_series: list = None, input_ids: Tensor = None, input_texts: list = None, gt_ts: list = None, mode: str = 'forecast'):
        if mode == 'forecast':
            if time_series is None or input_ids is None:
                raise ValueError("For 'forecast' mode, time_series and input_ids must be provided")
            return self.forecast(time_series, input_ids, gt_ts)
        elif mode == 'analyze':
            if time_series is None or input_ids is None:
                raise ValueError("For 'analyze' mode, time_series and input_ids must be provided")
            return self.analyze(time_series, input_ids)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'forecast', 'analyze', and 'generate'.")

    def forecast(self, time_series: list, input_ids: Tensor, gt_ts: list = None):
        """
        Inputs:
            time_series: list of T x C tensors with variable lengths (similar to analyze method)
            input_ids: Tensor, shape (Batch, Seq_Len)
            gt_ts: list of T' x C tensors with variable lengths (used to select appropriate pred_len for each sample)
        Outputs:
            dec_out_denorm: list of (Selected_Pred_Len, Channels) tensors for each sample
            dec_out: list of (Selected_Pred_Len, Channels) tensors for each sample
        Note: Each sample may have different pred_len based on its corresponding gt_ts length
        """
        # Normalize each time series in the list
        normalized_time_series = []
        for ts, gt_ts_i in zip(time_series, gt_ts):
            # ts shape: (T, C)
            ts_norm = self.normalize_layers(ts.unsqueeze(0), 'norm', auto_mask=False).squeeze(0)  # (T, C)
            gt_ts_i_channel = gt_ts_i.shape[1]
            ts_norm = ts_norm.reshape(-1, gt_ts_i_channel) # (T*C, 1)
            normalized_time_series.append(ts_norm)

        # Get batch info
        B = len(time_series)
        C = normalized_time_series[0].shape[1]  # Assuming all have same number of channels
        
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # (num_tokens, d_llm)

        # IMPORTANT: To ensure gradient synchronization in multi-node training,
        # we need to use all patch embedding modules and output projections in every forward pass
        # even if their outputs are not used. This prevents NCCL timeout issues.
        dummy_loss = 0.0
        dummy_input = torch.zeros(1, 1, 100, device=normalized_time_series[0].device, dtype=torch.bfloat16)
        # Dummy forward for analysis patch embeddings
        for patch_len_str, patch_embedding in self.analysis_patch_embeddings.items():
            dummy_out, _ = patch_embedding(dummy_input)
            dummy_loss = dummy_loss + dummy_out.sum() * 0.0  # Multiply by 0 to not affect gradients

        # Process each time series in the list with appropriate patch embedding
        enc_outs = []
        for b, ts in enumerate(normalized_time_series):
            # ts shape: (T, C)
            T, C = ts.shape
            for c in range(C):
                # Get the time series for this channel
                ts_channel = ts[:, c:c+1]  # (T, 1)
                ts_channel = ts_channel.permute(1, 0).unsqueeze(0).contiguous()  # (1, 1, T)
                
                # Select appropriate patch embedding based on sequence length
                selected_patch_len = self.select_analysis_patch_embedding(T)
                selected_patch_embedding = self.analysis_patch_embeddings[str(selected_patch_len)]
                enc_out, _ = selected_patch_embedding(ts_channel.to(torch.bfloat16))  # (1, patch_nums, analysis_d_model)
                enc_outs.append(enc_out.squeeze(0))  # (patch_nums, analysis_d_model)

        # Pad sequences to handle different patch numbers - using left padding
        # Reverse sequences, pad, then reverse back to achieve left padding
        enc_outs_reversed = [torch.flip(seq, [0]) for seq in enc_outs]
        enc_out_padded_reversed = torch.nn.utils.rnn.pad_sequence(enc_outs_reversed, batch_first=True)  # (B*C, max_patch_nums, analysis_d_model)
        enc_out = torch.flip(enc_out_padded_reversed, [1])  # (B*C, max_patch_nums, analysis_d_model) - left padded
        enc_out = self.analysis_reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # (B*C, patch_nums, d_llm)
        enc_out = enc_out + dummy_loss  # Add dummy loss to ensure all modules are in compute graph
        
        # Expand prompt_embeddings to match enc_out batch dimension
        prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids)  # (B, prompt_tokens, d_llm)
        prompt_embeddings = prompt_embeddings.repeat_interleave(C, dim=0)  # (B*C, prompt_tokens, d_llm)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        # Handle different model output types
        model_output = self.llm_model(inputs_embeds=llama_enc_out)
        if hasattr(model_output, 'last_hidden_state'):
            # For LlamaModel, GPT2Model, BertModel
            dec_out = model_output.last_hidden_state
        elif hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
            # For AutoModelForCausalLM (like Qwen3)
            dec_out = model_output.hidden_states[-1]  # Get the last layer's hidden states
        else:
            # Fallback - this shouldn't happen with properly configured models
            raise AttributeError(f"Model output type {type(model_output)} not supported")
        
        dec_out = dec_out[:, :, :self.d_ff] # (B*C, max_patch_nums + prompt_tokens, d_ff)

        dec_out = torch.reshape(
            dec_out, (-1, C, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous() # (B, C, d_ff, max_patch_nums + prompt_tokens)

        # Dummy forward for all output projections to ensure gradient synchronization
        dummy_loss = 0.0
        dummy_output_input = torch.zeros(1, 1, self.head_nf, device=normalized_time_series[0].device, dtype=torch.bfloat16)
        for pred_len_str, output_projection in self.output_projections.items():
            dummy_out = output_projection(dummy_output_input)
            dummy_loss = dummy_loss + dummy_out.sum() * 0.0  # Multiply by 0 to not affect gradients

        # Process each sample individually to select optimal pred_len
        dec_out_list = []
        dec_out_denorm_list = []
        
        for b in range(B):
            # Determine the appropriate pred_len for this specific sample
            if gt_ts is not None:
                gt_ts_b = gt_ts[b]
                gt_length = gt_ts_b.shape[0] * gt_ts_b.shape[1]  # T' * C
                ori_c = gt_ts_b.shape[1]
                selected_pred_len = self.select_pred_len(gt_length)
            else:
                # Default to the first pred_len if no gt_ts provided for this sample
                selected_pred_len = self.pred_lens[-1]
                ori_c = 1
            
            # Get the corresponding output projection for this sample
            selected_output_projection = self.output_projections[str(selected_pred_len)]
            
            # Apply output projection to this specific sample
            sample_dec_out = dec_out[b:b+1, :, :, -self.patch_nums:]  # (1, C, d_ff, patch_nums)
            sample_output = selected_output_projection(sample_dec_out)  # (1, C, selected_pred_len)
            sample_output += dummy_loss  # Add dummy loss to ensure all modules are in compute graph
            sample_output = sample_output.permute(0, 2, 1).contiguous()  # (1, selected_pred_len, C)
            sample_output = sample_output[:, -gt_length:, :]
            
            # Denormalize this sample
            sample_output = sample_output.reshape(1, -1, ori_c)  # (1, gt_len, C)
            sample_denorm = self.normalize_layers(sample_output, 'denorm', auto_mask=False)  # (1, gt_len, C)
            
            dec_out_list.append(sample_output.squeeze(0))  # (gt_len, C)
            dec_out_denorm_list.append(sample_denorm.squeeze(0))  # (gt_len, C)
        
        # Note: dec_out_list and dec_out_denorm_list contain tensors with potentially different pred_lens
        # For compatibility, we return the lists instead of stacked tensors
        return dec_out_denorm_list, dec_out_list

    def analyze(self, audios: list, input_ids: Tensor):
        """
        Analyze the audio and input_ids to get the labels.
        Inputs:
            audios: list of T x C tensors with variable lengths
            input_ids: (batch, input_len)
            labels: (batch, label_len)
        Returns:
            text: (batch, label_len)
        """
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) # (num_tokens, d_llm)
        
        # IMPORTANT: To ensure gradient synchronization in multi-node training,
        # we need to use all patch embedding modules in every forward pass
        # even if their outputs are not used. This prevents NCCL timeout issues.
        dummy_loss = 0.0
        dummy_input = torch.zeros(1, 1, 100, device=audios[0].device, dtype=torch.bfloat16)
        for patch_len_str, patch_embedding in self.analysis_patch_embeddings.items():
            dummy_out, _ = patch_embedding(dummy_input)
            dummy_loss = dummy_loss + dummy_out.sum() * 0.0  # Multiply by 0 to not affect gradients
        
        # Process each audio in the list
        audio_enc_outs = []
        for audio in audios:
            # audio shape: (T, C), assuming C=1
            T, C = audio.shape
            assert C == 1, "Expected input with 1 channel"
            audio = audio.permute(1, 0).unsqueeze(0).contiguous()  # (1, 1, T)
            
            # Select appropriate patch embedding based on audio length
            selected_patch_len = self.select_analysis_patch_embedding(T)
            selected_patch_embedding = self.analysis_patch_embeddings[str(selected_patch_len)]
            
            # Process audio with selected patch embedding
            audio_enc_out, n_vars = selected_patch_embedding(audio.to(torch.bfloat16)) # (1, patch_nums, d_model)
            audio_enc_outs.append(audio_enc_out.squeeze(0))

        audio_enc_out = torch.nn.utils.rnn.pad_sequence(audio_enc_outs, batch_first=True)  # (B, max_patch_nums, d_model)
        audio_enc_out = audio_enc_out + dummy_loss  # Add dummy loss to ensure all modules are in compute graph
        audio_enc_out = self.analysis_reprogramming_layer(audio_enc_out, source_embeddings, source_embeddings) # (B, max_patch_nums, d_llm)

        # Add time series start and end tokens
        batch_size = audio_enc_out.size(0)  # This is B*C
        ts_start_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.TS_START_TOKEN)], device=audio_enc_out.device).repeat(batch_size, 1)
        ts_end_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.TS_END_TOKEN)], device=audio_enc_out.device).repeat(batch_size, 1)
        
        ts_start_embeddings = self.llm_model.get_input_embeddings()(ts_start_ids)  # (B*C, 1, d_llm)
        ts_end_embeddings = self.llm_model.get_input_embeddings()(ts_end_ids)  # (B*C, 1, d_llm)
        
        # Wrap audio_enc_out with TS tokens: <ts> audio_enc_out </ts>
        audio_enc_out = torch.cat([ts_start_embeddings, audio_enc_out, ts_end_embeddings], dim=1) # (B*C, 1 + patch_nums + 1, d_llm)
        
        input_ids_embeddings = self.llm_model.get_input_embeddings()(input_ids)  # (B, input_len, d_llm)
        # Expand input_ids_embeddings to match audio_enc_out batch dimension (B*C)
        input_ids_embeddings = input_ids_embeddings.repeat_interleave(C, dim=0)  # (B*C, input_len, d_llm)
        audio_enc_out = torch.cat([audio_enc_out, input_ids_embeddings], dim=1) # (B*C, 2 + patch_nums + input_len, d_llm)
        
        # For causal LLM, we don't need to explicitly pass attention mask
        # The model will automatically apply causal masking
        
        model_output = self.llm_model(inputs_embeds=audio_enc_out)
        
        return model_output

    @torch.no_grad()
    def generate(self, audios: list, prompt: str = None, generation_config: dict = {}):
        """
        Generate text based on audio input (supports variable length audio sequences).
        Args:
            audios: list of T x C tensors with variable lengths (following analyze method format)
            prompt: Optional text prompt to guide generation (string or list of strings for batch)
            generation_config: Optional dictionary with generation parameters
        Returns:
            Generated text string (for single input) or list of strings (for batch input)
        """
        # Process each audio in the list using the same logic as analyze()
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (num_tokens, d_llm)
        
        # Process each audio in the list
        audio_enc_outs = []
        for audio in audios:
            # audio shape: (T, C), assuming C=1
            T, C = audio.shape
            assert C == 1, "Expected input with 1 channel"
            audio = audio.permute(1, 0).unsqueeze(0).contiguous()  # (1, 1, T)
            
            # Select appropriate patch embedding based on audio length
            selected_patch_len = self.select_analysis_patch_embedding(T)
            selected_patch_embedding = self.analysis_patch_embeddings[str(selected_patch_len)]
            
            # Process audio with selected patch embedding
            audio_enc_out, n_vars = selected_patch_embedding(audio.to(torch.bfloat16)) # (1, patch_nums, d_model)
            audio_enc_outs.append(audio_enc_out.squeeze(0))
        
        audio_enc_out = torch.nn.utils.rnn.pad_sequence(audio_enc_outs, batch_first=True)  # (B, max_patch_nums, d_model)
        audio_enc_out = self.analysis_reprogramming_layer(audio_enc_out, source_embeddings, source_embeddings) # (B, max_patch_nums, d_llm)

        original_batch_size = audio_enc_out.size(0)
        expanded_batch_size = original_batch_size * C

        # Add time series start and end tokens
        ts_start_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.TS_START_TOKEN)], device=audio_enc_out.device).repeat(audio_enc_out.size(0), 1)
        ts_end_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.TS_END_TOKEN)], device=audio_enc_out.device).repeat(audio_enc_out.size(0), 1)
        
        ts_start_embeddings = self.llm_model.get_input_embeddings()(ts_start_ids)  # (B*C, 1, d_llm)
        ts_end_embeddings = self.llm_model.get_input_embeddings()(ts_end_ids)  # (B*C, 1, d_llm)
        
        # Wrap audio_enc_out with TS tokens: <ts> audio_enc_out </ts>
        audio_enc_out = torch.cat([ts_start_embeddings, audio_enc_out, ts_end_embeddings], dim=1)  # (B*C, 1 + patch_nums + 1, d_llm)
        
        if prompt is not None:
            if isinstance(prompt, str):
                # Single prompt for all original samples, repeat for each channel
                prompts = [prompt] * expanded_batch_size  # Repeat for B*C
            elif isinstance(prompt, list):
                # Batch prompts - should match original batch size (B)
                if len(prompt) != original_batch_size:
                    raise ValueError(f"Number of prompts ({len(prompt)}) must match original batch size ({original_batch_size})")
                # Expand prompts to match B*C: each prompt repeated C times
                prompts = []
                for p in prompt:
                    prompts.extend([p] * n_vars)  # n_vars is C (number of channels)
            else:
                raise TypeError("Prompt must be a string or list of strings")
            
            prompt_list = []
            for p in prompts:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_list.append(prompt)
            prompt_ids = self.tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(audio_enc_out.device)  # (B*C, prompt_len)
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)  # (B*C, prompt_len, d_llm)

            inputs_embeds = torch.cat([audio_enc_out, prompt_embeddings], dim=1)  # (B*C, audio_tokens + prompt_len, d_llm)
        else:
            inputs_embeds = audio_enc_out
        
        if hasattr(self.llm_model, 'generate'):
            generated_ids = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                **generation_config,
            )
            
            if hasattr(generated_ids, 'sequences'):
                generated_token_ids = generated_ids.sequences
            else:
                generated_token_ids = generated_ids
            
            generated_text = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            
            return generated_text
        else:
            raise NotImplementedError("The loaded LLM model does not support text generation. Please use a generative model.")

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def select_analysis_patch_embedding(self, audio_length):
        """
        Select appropriate analysis patch embedding based on audio length.
        patch_len list为1,2,4,8,16... 确保了ts_tokens在100-200之间
        """
        target_patch_len = audio_length / self.ts_tokens
        
        # 从 analysis_patch_lens 中选择比 target_patch_len 小的值中最大的那个
        valid_patch_lens = [pl for pl in self.analysis_patch_lens if pl <= target_patch_len]
        
        if valid_patch_lens:
            selected_patch_len = max(valid_patch_lens)
        else:
            # 如果没有比 target_patch_len 小的值，选择最小的那个
            selected_patch_len = min(self.analysis_patch_lens)
        
        return selected_patch_len

    def select_pred_len(self, gt_length):
        """
        Select the smallest pred_len that is greater than or equal to gt_ts length.
        Args:
            gt_length: The length of ground truth time series
        Returns:
            The smallest pred_len that is >= gt_length
        """
        if gt_length is None:
            return max(self.pred_lens)  # Default to the largest pred_len if gt_length is not provided
        
        # Find pred_lens that are greater than or equal to gt_length
        valid_pred_lens = [pl for pl in self.pred_lens if pl >= gt_length]
        
        if valid_pred_lens:
            # Return the smallest pred_len that is >= gt_length
            selected_pred_len = min(valid_pred_lens)
        else:
            # If no pred_len is >= gt_length, return the largest available pred_len
            selected_pred_len = max(self.pred_lens)
        
        return selected_pred_len

    @staticmethod
    def load_checkpoint(checkpoint_path, config=None, config_path=None):
        """
        Args:
            checkpoint_path: format: exp/setting/timestamp/epoch_x/pytorch_model/mp_rank_00_model_states.pt
        Returns:
            Model instance with loaded parameters
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if config is None:
            if config_path is None:
                base_dir = '/'.join(checkpoint_path.split('/')[:-3])
                config_path = f"{base_dir}/config.json"
            print(f"Without config input, auto load from {config_path}")
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            config = dotdict(config_data)

        # Check the dtype of the saved weights to determine model dtype
        sample_weight = next(iter(checkpoint["module"].values()))
        if sample_weight.dtype == torch.bfloat16:
            print(f"Loading checkpoint with bfloat16 precision")
            model = Model(config).bfloat16()
        elif sample_weight.dtype == torch.float16:
            print(f"Loading checkpoint with float16 precision")
            model = Model(config).half()
        else:
            print(f"Loading checkpoint with float32 precision")
            model = Model(config).float()
            
        model.load_state_dict(checkpoint["module"], strict=False)
        
        print(f"Loaded parameters (excluding frozen LLM parameters) from {checkpoint_path}")
        
        return model


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1, use_mlp=False):
        '''
        Reprogramming layer with two modes:
        - Attention-based (default): Maps time series patches to LLM space using cross-attention
        - MLP-based (ablation): Simple MLP transformation for comparison
        
        Args:
            d_model: Dimension of input embeddings (time series patches)
            n_heads: Number of attention heads (only used in attention mode)
            d_keys: Dimension of keys (only used in attention mode)
            d_llm: Dimension of LLM embeddings (output dimension)
            attention_dropout: Dropout rate
            use_mlp: If True, use MLP-based reprogramming; otherwise use attention-based
        '''
        super(ReprogrammingLayer, self).__init__()
        
        self.use_mlp = use_mlp
        
        if use_mlp:
            # MLP-based reprogramming for ablation experiment
            hidden_dim = d_model * 2  # Use 2x hidden dimension for expressiveness
            
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(attention_dropout),
                nn.Linear(hidden_dim, d_llm),
                nn.Dropout(attention_dropout)
            )
        else:
            # Original attention-based reprogramming
            # 重编程层本质上是在问："这个时间序列patch最像LLM词汇表中的哪些词？"
            # 然后将相似的词的表示加权组合，得到在LLM语义空间中的等价表示。
            d_keys = d_keys or (d_model // n_heads)

            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
            self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
            self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
            self.n_heads = n_heads
            self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Args:
            target_embedding: (B, L, d_model) - time series patch embeddings
            source_embedding: (S, d_llm) - LLM vocabulary embeddings (only used in attention mode)
            value_embedding: (S, d_llm) - LLM vocabulary embeddings (only used in attention mode)
        Returns:
            out: (B, L, d_llm) - transformed embeddings in LLM space
        """
        if self.use_mlp:
            # MLP mode: simply apply MLP transformation
            # source_embedding and value_embedding are ignored
            return self.mlp(target_embedding)
        else:
            # Attention mode: use cross-attention mechanism
            B, L, _ = target_embedding.shape
            S, _ = source_embedding.shape
            H = self.n_heads

            target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
            source_embedding = self.key_projection(source_embedding).view(S, H, -1)
            value_embedding = self.value_projection(value_embedding).view(S, H, -1)

            out = self.reprogramming(target_embedding, source_embedding, value_embedding)

            out = out.reshape(B, L, -1)

            return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """Cross-attention based reprogramming (only used in attention mode)"""
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
