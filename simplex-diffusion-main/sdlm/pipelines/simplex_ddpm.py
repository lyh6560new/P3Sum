from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
import logging
import datetime
from inference.inference_utils import split_into_masked_and_unmasked
from rouge_score import rouge_scorer
import numpy as np
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
#use logging to record the generation process and save it to a file with the same name as the model and contain the date and time
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.FileHandler(f"logging/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.error('This should go to both console and file')
from inference.inference_utils import logits_projection
from utils import scale, self_condition_preds,convert_to_simplex
from termcolor import colored
from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data3/whr/lyh/ControllingPoliticalBias/ssd-lm/roberta-base")
def apply_controlling_drift(args, perturbed_batch_diralpha,context=None,mask=None):
    if args.decode_ctr_lr <= 0:
        args.ctr_loss = -1
        return perturbed_batch_diralpha

    if args.ctr_model is None:
        args.ctr_model = AutoModelForSequenceClassification.from_pretrained(args.ctr_model_name).to(perturbed_batch_diralpha.device)
    optimizing_label_index = args.ctr_opt_label_idx

    if args.ctr_model_name2 and args.ctr_model2 is None:
        args.ctr_model2 = AutoModelForSequenceClassification.from_pretrained(args.ctr_model_name2).to(perturbed_batch_diralpha.device)

    optimizing_label_index2 = args.ctr_opt_label_idx2 if (args.ctr_model_name2) else None
    with torch.enable_grad():
        if "Fact" in args.ctr_model_name:
            perturbed_inputs_diralpha_4ctr = perturbed_batch_diralpha.clone()
            #perturbed_inputs_diralpha = perturbed_batch_diralpha
            if args.ctr_model_name2:
                #print(perturbed_batch_diralpha.shape)
                perturbed_batch_diralpha_4ctr2 = perturbed_batch_diralpha.clone()
            #print(perturbed_inputs_diralpha_4ctr.shape)
            perturbed_inputs_diralpha_4ctr.requires_grad_()
            perturbed_inputs_simplex_4ctr = torch.nn.functional.softmax(perturbed_inputs_diralpha_4ctr, dim=-1)
            perturbed_inputs_embeds_4ctr = torch.nn.functional.linear(perturbed_inputs_simplex_4ctr, args.ctr_model.get_input_embeddings().weight.t())
            context = convert_to_simplex(context,5.0,args.vocab_size)
            context = torch.nn.functional.linear(context, args.ctr_model.get_input_embeddings().weight.t())
            #context = perturbed_inputs_embeds_4ctr[:,:args.uni_con]
            #TODO: change length
            #print(perturbed_inputs_embeds_4ctr.shape)
            summary = perturbed_inputs_embeds_4ctr[:,-100:]
            context = context[:,:512-100]
            concat = torch.cat([summary,context], dim=1)
            #print(concat.shape)
            ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model(inputs_embeds=concat).logits, dim=-1)[:,optimizing_label_index].mean()
            args.ctr_loss = ctr_loss
            ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_inputs_diralpha_4ctr)[0]
            #perturbed_inputs_diralpha_4ctr = perturbed_inputs_diralpha_4ctr + args.decode_ctr_lr * ctr_delta
            
            perturbed_batch_diralpha = perturbed_batch_diralpha + args.decode_ctr_lr * ctr_delta
            #perturbed_inputs_diralpha = perturbed_inputs_diralpha[:,args.uni_con:]
        else:
            perturbed_batch_diralpha_4ctr = perturbed_batch_diralpha.clone()
            perturbed_batch_diralpha_4ctr.requires_grad_()
            perturbed_batch_simplex_4ctr = torch.nn.functional.softmax(perturbed_batch_diralpha_4ctr, dim=-1)
            perturbed_batch_embeds_4ctr = torch.nn.functional.linear(perturbed_batch_simplex_4ctr, args.ctr_model.get_input_embeddings().weight.t())
            ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model(inputs_embeds=perturbed_batch_embeds_4ctr).logits, dim=-1)[:,optimizing_label_index].mean()
            args.ctr_loss = ctr_loss
            ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_batch_diralpha_4ctr)[0]
            perturbed_batch_diralpha = perturbed_batch_diralpha + args.decode_ctr_lr * ctr_delta # we use a fixed balancing factor in this work, which can be improved in the future
    if args.ctr_model_name2:
        with torch.enable_grad():
            #perturbed_batch_diralpha_4ctr = perturbed_inputs_diralpha_4ctr2
            perturbed_batch_diralpha_4ctr2.requires_grad_()
            perturbed_batch_simplex_4ctr2 = torch.nn.functional.softmax(perturbed_batch_diralpha_4ctr2, dim=-1)
            perturbed_batch_embeds_4ctr2 = torch.nn.functional.linear(perturbed_batch_simplex_4ctr2, args.ctr_model2.get_input_embeddings().weight.t())
            #print(perturbed_batch_embeds_4ctr2.shape)
            ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model2(inputs_embeds=perturbed_batch_embeds_4ctr2).logits, dim=-1)[:,optimizing_label_index2].mean()
            args.ctr_loss = ctr_loss
            ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_batch_diralpha_4ctr2)[0]
            perturbed_batch_diralpha = perturbed_batch_diralpha + args.decode_ctr_lr2 * ctr_delta
    return perturbed_batch_diralpha

@dataclass
class SimplexDiffusionPipelineOutput(BaseOutput):
    """
    Output class for simplex diffusion pipelines.
    Args:
        simplex (`np.ndarray`)
            numpy array showing the denoised simplex representation.
        logits (`np.ndarray`) final generated logits before applying the projection.
    """

    simplex: np.ndarray
    logits: np.ndarray
    loss: np.ndarray


class SimplexDDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        model: Model architecture to denoise the latents (encoded token ids).
        scheduler ([`SchedulerMixin`]): A scheduler to denoise the encoded latent.
        ctr_args: control arguments
            decode_ctr_lr: learning rate for controlling
            ctr_model_name: name of the model used for controlling
            ctr_opt_label_idx: index of the label to optimize for


    """

    def __init__(
        self,
        model,
        scheduler,
        simplex_value,
        top_p,
        sampling_type,
        is_conditional_generation,
        tokenizer,
        classifier_free_uncond_input,
        temperature,
        guidance_softmax_combination,
        ctr_args=None
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.simplex_value = simplex_value
        self.top_p = top_p
        self.sampling_type = sampling_type
        self.is_conditional_generation = is_conditional_generation
        self.tokenizer = tokenizer
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.temperature = temperature
        self.guidance_softmax_combination = guidance_softmax_combination
        self.ctr = ctr_args is not None
        if self.ctr:
            self.ctr_args = ctr_args

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1.0,
    ) -> Union[SimplexDiffusionPipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            seq_length: (`int`), sequence length for the generated samples.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            batch (`torch.FloatTensor`): batch of input data, mostly used in the conditional generation setting.
        Returns:
            [`~pipeline_utils.SimplexDiffusionPipelineOutput`]: returns the generated simplex.
        """
        # Classifier_free guidance works only in the conditional generation case.
        classifier_free_guidance = (
            guidance_scale > 1.0 and self.is_conditional_generation
        )
        """
        if classifier_free_guidance:
            # Makes unconditional input for max sequence length, later we truncate it.
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=seq_length, return_tensors="pt"
            ).to(self.device)
            # Converts this to a simplex (batch_size, max_seq, vocab_size)
            uncond_simplex = convert_to_simplex(uncond_input["input_ids"], self.simplex_value, self.model.config.vocab_size)
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.model,torch.nn.DataParallel):
             self.model = self.model.module
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            # TODO(rabeeh): is giving the length cheating for this setting?
            # Adapts the sequence length to the given `span_mask`'s length.
            seq_length = batch["input_ids"].shape[1]
        simplex_shape = (batch_size, seq_length, vocab_size)
        simplex = self.simplex_value * torch.randn(
            simplex_shape, generator=generator, device=self.device
        )
        if self.model.config.self_condition is not None:
            previous_pred = torch.zeros(
                (batch_size, seq_length, vocab_size), device=self.device
            )
        logits_projection_fct = lambda x: logits_projection(  # noqa: E731
            x, self.sampling_type, self.top_p, self.simplex_value, self.temperature
        )
        context_sequences = tokenizer.batch_decode(batch["input_ids"].detach().to('cpu'),skip_special_tokens=True)
        logger.info(f"context: {context_sequences}")
        for t in self.progress_bar(self.scheduler.timesteps):
            #print(t)
            # TODO(rabeeh): also check without the scale.
            t_scaled = scale(t, len(self.scheduler))
            """
            if classifier_free_guidance:
                if self.classifier_free_uncond_input == "empty_token":
                    uncond_input = uncond_simplex[:, : batch["input_ids"].shape[1], :]
                elif self.classifier_free_uncond_input == "noisy_simplex":
                    uncond_input = self.simplex_value * torch.randn(simplex.shape, generator=generator, device=self.device)
                else:
                    raise NotImplementedError
            """
            # 1. predict noise model_output. Note we need not to pass the input_ids in case of
            # unconditional generation since the loss would be computed and it should not.
            model_output = self.model(
                input_ids=batch["input_ids"]
                if self.is_conditional_generation
                else None,
                span_mask=batch["span_mask"]
                if self.is_conditional_generation
                else None,
                simplex=simplex,
                timesteps=t_scaled,
                previous_pred=previous_pred
                if self.model.config.self_condition
                else None,
                classifier_free_guidance=classifier_free_guidance,
                # unconditional_simplex=uncond_input if classifier_free_guidance else None,
            )
            model_output_logits = model_output.logits

            # Performs classifier-free guidance.
            if classifier_free_guidance:
                logits_uncond, logits_pred = model_output_logits.chunk(2)
                if self.guidance_softmax_combination:
                    model_output_logits = F.softmax(
                        logits_uncond, dim=-1
                    ) + guidance_scale * (
                        F.softmax(logits_pred, dim=-1)
                        - F.softmax(logits_uncond, dim=-1)
                    )
                else:
                    model_output_logits = logits_uncond + guidance_scale * (
                        logits_pred - logits_uncond
                    )
            

            if self.model.config.self_condition is not None:
                if classifier_free_guidance:
                    prev_output_logits = model_output.logits.chunk(2)[1]
                else:
                    prev_output_logits = model_output_logits

                previous_pred = self_condition_preds(
                    self.model.config.self_condition,
                    prev_output_logits,
                    logits_projection_fct,
                )
            #TODO: add control here
            # if (not self.ctr_args.ctr_model_name) or ( not "Fact" in self.ctr_args.ctr_model_name):
            #     equivalent_score = equivalent_score[:, unit_context_input_ids.size(1):].contiguous()
            #args.uni_con = unit_context_input_ids.size(1)
            if (self.ctr):
                context = torch.where(batch["span_mask"], batch["input_ids"], 0)
                model_output_logits = apply_controlling_drift(self.ctr_args, model_output_logits,context,batch["span_mask"])
            # Projection.
            projected_logits = logits_projection_fct(model_output_logits)

            # 2. compute previous logits: x_t -> x_t-1
            noise = self.simplex_value * torch.randn(
                simplex_shape, generator=generator, device=self.device
            )
            simplex = self.scheduler.step(
                projected_logits, t, noise, generator=generator
            ).prev_sample
            ###print 
            #print(batch)
            if t%100==0 or t==1 :
                #equivalent_score = projected_logits
                #logger.info(f"sigma_t={t_scaled}, training_coef_at_t={torch.sqrt(1 - alpha_t_bar)}")
                #logger.info(f"predicted simplex's entropy={torch.distributions.categorical.Categorical(logits=projected_logits).entropy()}, logit_max,min,mean={torch.max(equivalent_score)},{torch.min(equivalent_score)},{torch.mean(equivalent_score)}")

                
                
                
                unit_seq_len = batch["input_ids"].shape[1]
                # real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                # sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
                # logger.info(f"t={t}: {colored(str(sampled_sequences), 'red')}")



                simplex = projected_logits
                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                masked = list(
                            map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=True), real_token_ids_list.detach().to('cpu'), batch['span_mask'])
                            )
                labels = torch.where(batch["span_mask"], batch["input_ids"], 1)
                batch['gold'] = tokenizer.batch_decode(labels.detach().to('cpu'), skip_special_tokens=True)  

                #print(masked)
                #sampled_sequences = tokenizer.batch_decode(masked)
                sampled_sequences = [tokenizer.batch_decode(x, skip_special_tokens=True) for x in masked]
                logger.info(f"t={t} (before +z): {colored(str(sampled_sequences), 'green')}")


                # alt_i = 1 # look at the second best candidate
                # alt_real_token_ids_list = torch.topk(simplex, alt_i+1, dim=-1).indices[:, :, alt_i].view(batch_size, unit_seq_len)
                # alt_sampled_sequences = tokenizer.batch_decode(alt_real_token_ids_list.clone().detach().to('cpu'))
                # logger.info(f"t={t} (alt{alt_i+1}): {colored(str(alt_sampled_sequences), 'blue')}")

                #logger.info(f"ctr loss: {args.ctr_loss}")

                logger.info(f"non-zero vocab: {torch.count_nonzero(projected_logits > -5+0.0001) / simplex.size(0) / simplex.size(1)} out of {torch.numel(projected_logits) / simplex.size(0) / simplex.size(1)}")
                logger.info(f"ctr loss:{self.ctr_args.ctr_loss}")
                for a,b in zip(batch['gold'], sampled_sequences):
                    #print(a,b
                    a = str(a)
                    b = str(b)
                    logger.info("ROUGE-1: {:.4f}, ROUGE-2: {:.4f}, ROUGE-L: {:.4f}".format(
                    scorer.score(a, b)['rouge1'].fmeasure,
                    scorer.score(a, b)['rouge2'].fmeasure,
                    scorer.score(a, b)['rougeL'].fmeasure))

            if(t==0):
                logger.info(f"t={t}: {colored(str(batch['gold']), 'yellow')}")
        return SimplexDiffusionPipelineOutput(
            simplex=simplex, logits=model_output_logits, loss=model_output.loss
        )
