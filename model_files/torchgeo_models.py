# here, import torchgeo and instanciate trainer. Return trainer
# inputs: all the trainer configs, such as GPUs etc
# return: PyTorch Lightning Trainer object


def create_torchgeo_models(config):
    import timm
    from torchgeo.models import ResNet18_Weights
    
    # get model info
    bands = config.data.bands
    type = config.model.model_type
    classes = config.model.n_classes

    if "resnet" in type.lower():
        assert config.data.bands==3,"Model only uses RGB bands. Make sure the config specifies the correct number of input channels."
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
        model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"], num_classes=classes)
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
    elif "farseg" in type.lower():
        assert config.data.bands==3,"Model only uses RGB bands. Make sure the config specifies the correct number of input channels."
        from torchgeo.models import FarSeg
        model = FarSeg(backbone='resnet50', classes=classes, backbone_pretrained=True)
    elif "fcn" in type.lower():
        from torchgeo.models import FCN
        model = FCN(bands, classes, num_filters=64)
    # return
    return model

