# 03. GTSRB Transfer Learning with PyTorch

## 1. Transfer Learning Project Contract
    1.1 Problem statement
    1.2 Previous from-scratch benchmark
    1.3 Research questions
    1.4 Experiment rules
    1.5 Success criteria

## 2. Setup and Saved Split Loading
    2.1 Import libraries
    2.2 Set project paths
    2.3 Load helper files
    2.4 Load saved train/valid/test splits
    2.5 Device and seed setup
    2.6 Quick split sanity check

## 3. Transfer Learning Theory
    3.1 What is transfer learning?
    3.2 What is ImageNet pretraining?
    3.3 Feature extraction vs fine-tuning
    3.4 What does freezing mean?
    3.5 What does replacing classifier head mean?
    3.6 Why pretrained models need different transforms
    3.7 Why 224x224 image size is commonly used

## 4. Transfer Learning Data Pipeline
    4.1 Choose image size and normalization strategy
    4.2 Create training transform
    4.3 Create validation/test transform
    4.4 Recreate GTSRB Dataset class
    4.5 Create DataLoaders
    4.6 Visualize transformed images
    4.7 Check one batch

## 5. TL Model 0: ResNet18 Feature Extractor
    5.1 Why ResNet18 first?
    5.2 Load pretrained ResNet18 weights
    5.3 Inspect original classifier head
    5.4 Freeze all backbone parameters
    5.5 Replace final fc layer for 43 classes
    5.6 Check trainable parameters
    5.7 One-batch forward pass
    5.8 Define loss and optimizer
    5.9 Train classifier head only
    5.10 Save best checkpoint
    5.11 Plot training curves
    5.12 Evaluate validation set
    5.13 Observation and decision

## 6. TL Model 1: ResNet18 Fine-Tuning Layer4
    6.1 Why fine-tune only layer4?
    6.2 Load TL Model 0 best checkpoint
    6.3 Unfreeze layer4 and classifier head
    6.4 Check trainable parameters again
    6.5 Use different learning rates for backbone and classifier
    6.6 Train fine-tuned model
    6.7 Save best checkpoint
    6.8 Plot training curves
    6.9 Evaluate validation set
    6.10 Observation and decision

## 7. TL Model 2: Optional EfficientNet-B0 Feature Extractor
    7.1 Why test EfficientNet-B0?
    7.2 Load pretrained EfficientNet-B0
    7.3 Freeze features
    7.4 Replace classifier
    7.5 Train and evaluate
    7.6 Compare with ResNet18

## 8. Transfer Learning Experiment Comparison
    8.1 Compare TL models
    8.2 Compare with from-scratch Model 4
    8.3 Select final candidate model

## 9. Official Test Evaluation
    9.1 Load final best model
    9.2 Evaluate on official test set
    9.3 Classification report
    9.4 Confusion matrix
    9.5 Wrong prediction analysis

## 10. Final Summary and Next Steps
    10.1 What worked
    10.2 What did not work
    10.3 From-scratch vs transfer learning conclusion
    10.4 Future improvements