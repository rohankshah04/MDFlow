from model import MDFlow
from flow_matching import train_step_flow_matching

def main():
    # 1) Build config (from config.py)
    cfg = model_config(name="model_1", train=True)

    # 2) Instantiate model
    model = MDFlow(cfg, use_velocity_head=True).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 3) Create data loader from your ATLAS dataset
    #    e.g. each batch has:
    #      batch["coords_t"], batch["coords_next"], batch["t"], ...
    loader = make_dataloader_atlas(...)  

    # 4) Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(loader):
            loss_val = train_step_flow_matching(
                model, batch, optimizer,
                config=cfg, device="cuda",
                loss_mode="velocity",  # or "next_frame"
                add_noise=True
            )
            if step % 10 == 0:
                print(f"Epoch {epoch}, step {step}, loss {loss_val}")