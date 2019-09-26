## process v1
 - val set에 train_transform 

```python
# transform
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        
test_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        
# coef
coef = [0.57, 1.37, 2.57, 3.57] # 왜 1.37로 되어있는지...


# model (no dropout...)
model = get_efn_model(model_num=model_num, isPretrained=True)
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, config.NUM_CLASSES)

```
