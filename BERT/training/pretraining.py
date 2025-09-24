import torch
import torch.nn.functional as F
from tqdm import tqdm
from .optimization import get_optimizer_and_scheduler
from .utils import save_checkpoint

def pretrain_loop(model, dataloader, config, start_step=0):
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)

    global_step = start_step
    model.train()
    pbar = tqdm(total=config["num_train_steps"], initial=global_step)
    while global_step < config["num_train_steps"]:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            nsp_labels = batch["nsp_label"].to(device)

            mlm_logits, nsp_logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            # MLM loss: cross-entropy over vocab, only where mlm_labels != -100
            mlm_loss = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1), ignore_index=-100)
            nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)

            loss = mlm_loss + nsp_loss

            scheduler.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"Step {global_step} Loss: {loss.item():.4f}")

            if global_step % config.get("save_every", 2000) == 0:
                ckpt_path = f"checkpoints/bert_step_{global_step}.pt"
                save_checkpoint(model, scheduler, global_step, ckpt_path)
                print("Saved checkpoint to", ckpt_path)

            if global_step >= config["num_train_steps"]:
                break
    pbar.close()
    # final save
    save_checkpoint(model, scheduler, global_step, "checkpoints/bert_final.pt")
    print("Training finished. Model saved to checkpoints/bert_final.pt")
