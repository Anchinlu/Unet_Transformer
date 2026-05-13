import torch
from utils.utils import calculate_metrics

def trainer(args, model, train_loader, val_loader, criterion, optimizer, device):
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            preds = model(images)
            loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                
                _, dice = calculate_metrics(preds, masks)
                val_dice += dice
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        print(f"\n=== EPOCH {epoch+1} ===")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}\n")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), f'{args.model}_best_model.pth')
            print(f"Saved {args.model}_best_model.pth with Dice: {best_dice:.4f}\n")
