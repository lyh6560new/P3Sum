import logging
import os
import sys

import datasets
import transformers
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from arguments import get_args
from data.data_collator import SpanInfillingDataCollator
from data.data_utils import load_data, tokenize_data_new
from inference.inference_utils import evaluate_generation
from models import load_model
from schedulers import SimplexDDPMScheduler
from trainer import DiffusionTrainer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")

require_version(
    "datasets>=2.0.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


def get_compute_metrics(data_args, training_args, model_args):
    # Causal language model.
    causal_model = AutoModelForCausalLM.from_pretrained(
        model_args.autoregressive_eval_model
    )
    causal_model = causal_model.to(training_args.device)
    causal_tokenizer = AutoTokenizer.from_pretrained(
        model_args.autoregressive_eval_model
    )

    is_conditional_generation = data_args.conditional_generation is not None
    prefix_lm_eval = data_args.conditional_generation in [
        "prefix_lm",
        "ul2",
        "ul2_with_unconditional",
        "ul2_variable",
    ]
    compute_metrics = lambda results: evaluate_generation(  # noqa: E731
        results,
        data_args,
        causal_model,
        causal_tokenizer,
        is_conditional_generation,
        prefix_lm_eval=prefix_lm_eval,
        skip_special_tokens=data_args.skip_special_tokens,
        eval_for_all_metrics=training_args.eval_for_all_metrics,
    )
    return compute_metrics


def main():
    # parse args
    model_args, data_args, training_args, diffusion_args = get_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load model
    tokenizer, model = load_model(model_args, diffusion_args, logger)

    # init schedulers
    noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )
    inference_noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_inference_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )

    if data_args.tokenized_data_path:
        tokenized_datasets = load_from_disk(data_args.tokenized_data_path)
    else:
        raw_datasets = load_data(data_args, model_args)
        tokenized_datasets = tokenize_data_new(
            data_args, tokenizer, raw_datasets, training_args
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        # convert eval dataset to regular dataset
        if isinstance(eval_dataset, datasets.IterableDataset):
            def iterable_generator():
                for x in eval_dataset:
                    yield x
            eval_dataset = Dataset.from_generator(iterable_generator)
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits):
            return logits.argmax(dim=-1)

    # Data collator
    # TODO: fix lambda max_seq_length, extra_padding_ratio:
    pad_to_multiple_of_8 = (
        data_args.line_by_line
        and training_args.fp16
        and not data_args.pad_to_max_length
    )
    data_collator = lambda mode: SpanInfillingDataCollator(  # noqa: E731
        mode=mode,
        data_args=data_args,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        seed=training_args.seed,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        eval_context_size=data_args.eval_context_size,
    )

    if training_args.do_eval:
        compute_metrics = get_compute_metrics(data_args, training_args, model_args)

    if data_args.shuffle:
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    # Initialize our Trainer
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.do_eval and not training_args.without_compute_metrics
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
        noise_scheduler=noise_scheduler,
        diffusion_args=diffusion_args,
        data_args=data_args,
        inference_noise_scheduler=inference_noise_scheduler,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    #model_args.model_name_or_path = 
    if training_args.do_eval:
        if training_args.load_states_in_eval_from_model_path:
            trainer._load_from_checkpoint(model_args.model_name_or_path)
            trainer.state = TrainerState.load_from_json(
                os.path.join(model_args.model_name_or_path, "trainer_state.json")
            )
            trainer._load_rng_state(model_args.model_name_or_path)

        # np.save("weights.npy", model.vocab_to_hidden_dim_embed.weight.data.numpy())

        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
