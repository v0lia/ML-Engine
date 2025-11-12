from pathlib import Path

import torch

from src.data import get_data
from src.models import get_model
from src.loss import get_loss_function
from src.optimizer import get_optimizer
from src import train, evaluate
from src.utils import config_loader, argparser, seed, tensorboard_utils

def main():
    args = argparser.parse_args(config_loader.default_config_path)
    mode_bool = args.mode == "train"    # choices=["train", "eval", "evaluate"]
    
    config = config_loader.get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    seed.set_seeds(config, device)

    dataloader = get_data.get_dataloader(args.mode, config)
    model_name = Path(args.model.strip()).stem
    model = get_model.get_model_class(model_name).to(device)
    loss_fn = get_loss_function.get_loss_function(config, device)
    optimizer = get_optimizer.get_optimizer(model, config)

    if mode_bool:   # train mode
        print(f"Starting training model {model_name}...")
        writer = tensorboard_utils.get_writer(model_name, True)   # True for train mode
        train.train(dataloader, model, loss_fn, optimizer, config, device, writer)
        print(f"Finished training model {model_name}!")
    else:           # eval mode
        print(f"Starting evaluating model {model_name}...")
        writer = tensorboard_utils.get_writer(model_name, False)    # False for eval mode 
        evaluate.evaluate(dataloader, model, loss_fn, config, device, writer)
        print(f"Finished evaluating model {model_name}!")
    tensorboard_utils.close_writer(writer)

if __name__ == "__main__":
    main()

# 0) вернуть папку datasets в корень (и файл .gitkeep)
# 1) настроить .gitignore

# 4) утилита logger.py

# 5) утилита checkpoint.py
# 6) утилита timer.py



# 11) readme.md - всё что я посчитаю необходимым
# 12) # В README-файле проекта опиши: “Структура проекта, как запустить”, “где лог-файлы”, “где визуализации”.
# 13) научиться вставлять картинки в ридми
# 14) описать в ридми результаты - все красивые графики (это основа продажи моего профессионализма)
# 20) что имеется в виду под "можно добавить GitHub Actions, которые гоняют pytest при каждом PR → dev."
# 21) CI
# 30) autotests?
# 40 - эксперименты с моделью - сделать законченный проект
# 50 - выложить на гитхаб
# 51 - причесать фашнмнист на гитхабе

############################################

# 100 - цифар10 - новый проект
# 101 - цифар10 - правки в коде
# 102 - цифар10 - эксперименты с моделью
# 110 - цифар10 - выложить на гитхабе
# 111 - цифар10  причесать
# 120 - причесать мастер-портфолио ридми

############################################

# 201 - нлп модель - урок 1
# 202 - нлп модель - урок 2
# 203 - нлп модель - урок 3

# 210 - нлп модель - новый проект
# 211 - нлп модель - правки в коде
# 212 - нлп модель - эксперименты с моделью

# 220 - нлп модель - выложить на гитхабе
# 221 - нлп модель - причесать на гитзабе
# 230 - причесать мастер-портфолио ридми

# 250 - пройти A guide on good usage of non_blocking and pin_memory() in PyTorch
# 251 - имплементировать в проектах
# 255 - ?описать в ридми?
# 256 - залить на гитхаб

# 260 - пройти Visualizing Gradients
# 261 - имплементировать в проектах
# 265 - описать в ридми
# 266 - залить на гитхаб

# 300 - наметить дальнейший курс с ChatGPT/DeepSeek
# 301 - наметить дальнейший план