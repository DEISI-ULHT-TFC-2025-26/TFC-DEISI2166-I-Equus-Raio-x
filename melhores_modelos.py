from utils import *

def run(inicial = 0, treino_augmentation = 0, treino_augmentation_final = 0, treino_ln_opt = 0, treino_schedule = 0 ,treino_loss = 0):
    if inicial == 1:
        ###################################
        #   Treino inicial                #
        ###################################
        #Treino com dados "puros"
        path = "teste/dados_controlo"
        run_experiments(path, f"{path}/resultados_modelos_iniciais.csv", modelos_config, n_runs=1)
    


    if treino_augmentation == 1:
        ###################################
        #   Treino com data augmentation  #
        ###################################
        for i, augmentation in enumerate([training_transforms_1, training_transforms_2, training_transforms_3, training_transforms_4, training_transforms_5,training_transforms_6]):
            # 
            for modelo in list(modelos_config.keys()):
                path = f"teste/augmentation_{i+1}"
                
                config_copy = {modelo : copy.deepcopy(modelos_config[modelo])}
                config_copy[modelo]["train_transform"] = augmentation

                run_experiments(path, f"{path}/resultados_data_augmentation{i+1}.csv", config_copy, n_runs=1)


    if treino_augmentation_final == 1:
        ######################################
        # Treino com data augmentation final #
        ######################################
        for i, augmentation in enumerate([training_transforms_final1, training_transforms_final2, training_transforms_final3, training_transforms_final4]):
            for modelo in list(modelos_config.keys()):
                path = f"teste/augmentation_final{i+1}"

                config_copy = {modelo : copy.deepcopy(modelos_config[modelo])}
                config_copy[modelo]["train_transform"] = augmentation

                run_experiments(path, f"{path}/resultados_data_augmentation_final{i+1}.csv", config_copy, n_runs=1)
    


    if treino_ln_opt == 1:
        change_dictionary(variavel_tipo="augmentation")
        learning_rates = [0.01, 0.001]
        optimizer_config = {
            "default": ["AdamW", "SGD", "Adan"],
            "YOLO": ["AdamW", "SGD", "auto"]
        }
        
        for ln in learning_rates:
            for modelo in list(modelos_config.keys()):
                if modelo.startswith("YOLO"):
                    opts = optimizer_config["YOLO"]
                else:
                    opts = optimizer_config["default"]

                for opt in opts:
                    path = f"teste/optimizer_{opt}_{ln}"
                    config_copy = {modelo : copy.deepcopy(modelos_config[modelo])}

                    config_copy[modelo]["optimizer"] = opt
                    config_copy[modelo]["learning_rate"] = ln
                    run_experiments(path, f"{path}/resultados_{opt}_{ln}.csv", config_copy, n_runs=1)
        
    

    if treino_schedule == 1:
        change_dictionary(variavel_tipo="optimizer")

        schedule_config = {
            "default": ["None", "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"],
            "YOLO": ["None","CosineAnnealingLR"]
        }

        for modelo in modelos_config:
            if modelo.startswith("YOLO"):
                schedule = schedule_config["YOLO"]
            else:
                schedule = schedule_config["default"]

            for sch in schedule:
                path = f"teste/scheduler_{sch}"
                config_copy = {modelo : copy.deepcopy(modelos_config[modelo])}
                config_copy[modelo]["scheduler"] = sch

                run_experiments(path, f"{path}/resultados_{sch}.csv", config_copy, n_runs=1)
    


    if treino_loss == 1:
        change_dictionary(variavel_tipo="scheduler")

        loss_config = {
            "default": ["crossEntropy", "crossEntropy+smooth", "kldiv", "mse"],
        }

        for modelo in modelos_config:
            if modelo.startswith("YOLO"):
                continue
            
            loss_function = loss_config["default"]

            for loss_f in loss_function:
                path = f"teste/lossFunction_{loss_f}"
                config_copy = {modelo : copy.deepcopy(modelos_config[modelo])}
                config_copy[modelo]["loss_type"] = loss_f
                run_experiments(path, f"{path}/{loss_f}.csv", config_copy, n_runs=1)
            



if __name__ == "__main__":
    freeze_support()

    run(inicial = 1, treino_augmentation = 1, treino_augmentation_final = 0, treino_ln_opt = 1, treino_schedule = 1, treino_loss = 0)
