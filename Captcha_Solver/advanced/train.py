def train():
    device = torch.device(config.DEVICE)
    model = AdvancedCaptchaModel().to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    print("\n" + "="*50)
    print(f"🎓 LAB COMMENCED | Device: {config.DEVICE}")
    print(f"📡 Plan: {STAGE_1_EPOCHS} Digits -> {STAGE_2_EPOCHS} Hybrid")
    print("="*50)

    dataset = CurriculumDataset(mode='digits')
    
    for epoch in range(1, STAGE_1_EPOCHS + STAGE_2_EPOCHS + 1):
        if epoch == STAGE_1_EPOCHS + 1:
            print("\n" + "!"*50)
            print("🚀 CURRICULUM UPGRADE: Switching to Full Alphanumeric + Hybrid")
            print("!"*50)
            dataset = CurriculumDataset(mode='full')
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005 
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        model.train()
        
        epoch_loss = 0
        correct, total = 0, 0
        start_time = time.time()
        
        num_batches = len(dataloader)
        
        for batch_idx, (imgs, target) in enumerate(dataloader):
            imgs, target = imgs.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Internal Accuracy Tracking
            out_reshaped = output.view(-1, 6, 62).argmax(2)
            tar_reshaped = target.view(-1, 6, 62).argmax(2)
            correct += (out_reshaped == tar_reshaped).sum().item()
            total += tar_reshaped.numel()

            # --- LIVE PROGRESS UPDATE ---
            if batch_idx % 10 == 0:
                current_acc = (correct / total) * 100
                print(f"\r  ⚡ [Epoch {epoch:02d}] Batch {batch_idx:03d}/{num_batches} | Loss: {loss.item():.4f} | Acc: {current_acc:.1f}%", end="")

        avg_loss = epoch_loss / num_batches
        final_acc = (correct / total) * 100
        
        # --- EPOCH SUMMARY ---
        print(f"\n✅ Epoch [{epoch:02d}] Complete | Phase: {'DIGITS' if epoch <= STAGE_1_EPOCHS else 'HYBRID'}")
        print(f"   Total Loss: {avg_loss:.4f} | Avg Acc: {final_acc:.2f}% | Time: {time.time()-start_time:.1f}s")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "advanced_lab_model.pth")
            print(f"   💾 Checkpoint saved.")
