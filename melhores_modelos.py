from utils import *

def run(inicial = 0, treino_augmentation = 0, treino_augmentation_final = 0):
    path = "treino_modelos"
    if inicial == 1:
        ###################################
        #   Treino inicial                #
        ###################################
        #Treino com dados "puros"
        run_experiments(f"{path}/resultados_modelos_iniciais.csv", modelos_config, transforms_0, transforms_0, path, n_runs=2)
    

    if treino_augmentation == 1:
        ###################################
        #   Treino com data augmentation  #
        ###################################
        for i, augmentation in enumerate([training_transforms_1, training_transforms_2, training_transforms_3, training_transforms_4, training_transforms_5, training_transforms_6]):
            run_experiments(f"{path}/resultados_data_augmentation{i+1}.csv", modelos_config, augmentation, transforms_0, path, n_runs=2)


    if treino_augmentation_final == 1:
        ######################################
        # Treino com data augmentation final #
        ######################################
        #Treino com aug finais
        for i, augmentation in enumerate([training_transforms_final1, training_transforms_final2, training_transforms_final3, training_transforms_final4]):
            path = f"aug_f{i+1}"
            run_experiments(f"{path}/resultados_data_augmentation_final{i+1}.csv", modelos_config, augmentation, transforms_0, path, n_runs=2)


run(inicial = 0, treino_augmentation = 0, treino_augmentation_final = 0)