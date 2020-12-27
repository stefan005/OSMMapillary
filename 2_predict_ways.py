from os import walk

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def predict(model, transform_function, image_dir) -> list:
    (_, _, filenames) = next(walk(image_dir))
    if len(filenames) > 5:
        probabilities = 0
        for f_path in filenames:
            image = Image.open(image_dir + '/' + f_path)
            transformed = transform_function(image).float().unsqueeze(0)
            predicted = model(transformed)
            probabilities += F.softmax(predicted, dim=1)
        probabilities = probabilities/len(filenames)
    else:
        probabilities = None
    return probabilities, len(filenames)

def main() -> None:
    #TODO add argument parser
    model = torch.load("test-model-2020-12-26.pth")
    model.eval()
    classes = ['asphalt', 'cobblestone', 'concrete', 'fine_gravel', 'grass', 'ground', 'paving_stones', 'sett']
    data_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                                            ])
    (_, dirname, _) = next(walk("./ressources/mapillary_raw/unlabeled")) 
    result = dict()
    for way_id in dirname:
        probability, cnt_images = predict(model, data_transform, "./ressources/mapillary_raw/unlabeled/"+way_id)
        if probability != None:
            ps = torch.tensor(probability)
            top_p, top_class = probability.topk(1, dim=1)
            result[way_id] = (classes[top_class], top_p.detach().numpy()[0][0],cnt_images)
            print(f"way_id: {way_id} "
                f"probability: {np.round(probability.detach().numpy(),4)[0]}\n"
                f"top_class: {classes[top_class]} with probablity {np.round(top_p.detach().numpy(),4)[0][0]}"
                f"with {cnt_images} images\n\n")
        else:
            result[way_id] = None
    df = pd.DataFrame(result,["pred_surface","probability","cnt_images"])
    print(df.T)
    print(result)


if __name__ == "__main__":
    main()