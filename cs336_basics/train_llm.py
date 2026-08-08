from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
import numpy as np
from cs336_basics.utils import lr_cosine_schedule, get_batch, save_checkpoint, load_checkpoint, gradient_clipping, cross_entropy
import wandb
import argparse
import torch
import time
parser = argparse.ArgumentParser(description="Train an LLM")

# Model arguments
parser.add_argument("--vocab-size", type=int, default=10_000)
parser.add_argument("--context-length", type=int, default=256)
parser.add_argument("--num-layers", type=int, default=4)
parser.add_argument("--d-model", type=int, default=512)
parser.add_argument("--num-heads", type=int, default=16)
parser.add_argument("--d-ff", type=int, default=1344)
parser.add_argument("--theta", type=float, default=10000)
parser.add_argument("--max-seq-len", type=int, default=256)

# Optimizer arguments
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--max-learning-rate", type=float, default=3e-4)
parser.add_argument("--min-learning-rate", type=float, default=3e-5)
parser.add_argument("--warmup-iters", type=int, default=1)
parser.add_argument("--cosine-cycle-iters", type=int, default=100)

parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--max-l2-norm", type=float, default=1.0)


parser.add_argument("--train-path", type=str)
parser.add_argument("--valid-path", type=str)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--valid-batch-size", type=int, default=1)
parser.add_argument("--num-val-batches", type=int, default=1)

parser.add_argument("--device", type=str)
parser.add_argument("--model-path", type=str)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from a checkpoint",
)
parser.add_argument("--num-iters", type=int, default=100)
parser.add_argument("--eval-steps", type=int, default=10)
parser.add_argument("--save-steps", type=int, default=10)





def main():
    args = parser.parse_args()
    torch.manual_seed(42)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        max_seq_len=args.max_seq_len,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    model.to(args.device)

    if args.resume is True:
        index = load_checkpoint(src=args.model_path, model=model, optimizer=optimizer)+1
    else:
        index = 0
    train_rng = np.random.default_rng(42)
    val_rng = np.random.default_rng(42)

    train_dataset = np.load(args.train_path, mmap_mode='r')
    valid_dataset = np.load(args.valid_path, mmap_mode='r')

    with wandb.init() as run:
        start = time.perf_counter()
        for t in range(index,args.num_iters):
            model.train()
            inputs, targets = get_batch(train_rng,dataset=train_dataset,batch_size=args.batch_size,context_length=args.context_length,device=args.device)
            optimizer.zero_grad()  
            token_positions = torch.arange(0,args.context_length,device=args.device)
            logits = model(inputs, token_positions)
            loss = cross_entropy(logits,targets)
            loss.backward()
            lr =  lr_cosine_schedule(t, max_learning_rate=args.max_learning_rate, min_learning_rate=args.min_learning_rate, warmup_iters=args.warmup_iters, cosine_cycle_iters=args.cosine_cycle_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr

            gradient_clipping(model.parameters(), max_l2_norm=args.max_l2_norm)
            optimizer.step()
            if t%args.eval_steps == 0:
                val_rng = np.random.default_rng(42)

                model.eval()
                with torch.no_grad():
                    total_val_loss = 0
                    for i in range(args.num_val_batches):
                        inputs, targets = get_batch(val_rng, dataset=valid_dataset,batch_size=args.valid_batch_size,context_length=args.context_length,device=args.device)
                        logits = model(inputs, token_positions)
                        total_val_loss += cross_entropy(logits, targets)
                    run.log({"elapsed_seconds":time.perf_counter()-start,"lr":lr,"loss":loss, "val_loss":total_val_loss/args.num_val_batches,"tokens_processed":(t+1)*args.batch_size*args.context_length},step=t)
            else:
                run.log({"elapsed_seconds":time.perf_counter()-start,"lr":lr,"loss":loss,"tokens_processed":(t+1)*args.batch_size*args.context_length}, step=t)

            if t%args.save_steps == 0:
                save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=args.model_path)
        save_checkpoint(model=model, optimizer=optimizer, iteration=t, out=args.model_path)


if __name__ == "__main__":
    main()