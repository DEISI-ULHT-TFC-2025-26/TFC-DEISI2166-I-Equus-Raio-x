from imports import *
###################################
#   Funcoes de Data               #
###################################
def split_data():
    ########################################################
    #   Funcao para dividir a data em 80/20 treino e teste #
    ########################################################
    original = "dados"
    base = "dados_split"

    for split in ["train", "test"]:
        for cls in os.listdir(original):
            os.makedirs(os.path.join(base, split, cls), exist_ok=True)

    for classe in os.listdir(original): 
        classe_path = os.path.join(original, classe) 
        imagens = os.listdir(classe_path) 

        imagens = [os.path.join(classe_path, img) for img in imagens if img.lower().endswith((".jpg", ".jpeg", ".png"))] #Lista com todas as imagens associada a sua classe

        treino, teste = train_test_split(imagens, test_size = 0.2, random_state=42)

        for img_path in treino:
            dest = os.path.join(base, "train", classe, os.path.basename(img_path))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(img_path, dest)
            
        for img_path in teste:
            dest = os.path.join(base, "test", classe, os.path.basename(img_path))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(img_path, dest)



def img_process(train_transforms, test_transforms):
    #################################################################
    #   Funcao que aplica os metodos de data augmentation nos dados #
    #################################################################
    test_data = datasets.ImageFolder('dados_split2/test', transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    train_data = datasets.ImageFolder('dados_split2/train', transform=train_transforms)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    return test_loader, train_loader


###################################
#   Funcoes de treino             #
###################################
def best_epoch(newpath, nome, run, model, optimizer, schedule, loss_type, train_loader, test_loader):
    #####################################################################################
    #   Funcao que treina e testa um modelo a cada epoch para achar o melhor valor de f1 #
    #####################################################################################
    
    if loss_type == "crossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "crossEntropy+smooth":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif loss_type == "kldiv":
        criterion = nn.KLDivLoss(reduction="batchmean")
    elif loss_type == "mse":
        criterion = nn.MSELoss()

    if schedule == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif schedule == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    elif schedule == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        scheduler = None


    best_f1 = -1
    best_epoch_num = 0
    historico_f1 = []
    paciencia = 30

    #Utilizar CPU ou cuda
    model.to(device)
    

    #Treinar e avaliar o modelo
    for epoch in range(1, 200 +1):
        model.train()
        running_loss = 0.0
        percentagem = int((epoch/200) * 100) 

        #Treinar o modelo
        for imagens, labels in train_loader:
            imagens = imagens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imagens)

            if loss_type in ["crossEntropy", "crossEntropy+smooth"]:

                loss = criterion(outputs, labels)

            elif loss_type == "kldiv":

                # logits -> log probabilities
                log_probs = F.log_softmax(outputs, dim=1)

                # labels -> one hot
                targets = F.one_hot(
                    labels,
                    num_classes=outputs.shape[1]
                ).float()

                loss = criterion(log_probs, targets)

            elif loss_type == "mse":

                probs = F.softmax(outputs, dim=1)

                targets = F.one_hot(
                    labels,
                    num_classes=outputs.shape[1]
                ).float()

                loss = criterion(probs, targets)
            
            #loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        
        #Avaliar o modelo
        model.eval()
        val_loss = 0.0
        previsoes = []
        real = []

        with torch.no_grad():
            for imagens, labels in test_loader:
                imagens, labels = imagens.to(device), labels.to(device)
                outputs = model(imagens)

                if loss_type in ["crossEntropy", "crossEntropy+smooth"]:

                    loss = criterion(outputs, labels)

                elif loss_type == "kldiv":

                    log_probs = F.log_softmax(outputs, dim=1)

                    targets = F.one_hot(
                        labels,
                        num_classes=outputs.shape[1]
                    ).float()

                    loss = criterion(log_probs, targets)

                elif loss_type == "mse":

                    probs = F.softmax(outputs, dim=1)

                    targets = F.one_hot(
                        labels,
                        num_classes=outputs.shape[1]
                    ).float()

                    loss = criterion(probs, targets)

                #loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1) 
                
                previsoes.extend(predicted.cpu().numpy()) 
                real.extend(labels.cpu().numpy())

        val_loss /= len(test_loader)
        
        if schedule == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        elif schedule in ["StepLR", "CosineAnnealingLR"]:
            scheduler.step()
        else:
            pass
        

        #calcular metrica
        f1 = f1_score(real, previsoes, average="macro") 
        historico_f1.append(f1)

        if f1 >= best_f1:
            best_f1 = f1
            best_epoch_num = epoch
            best_prev, best_real = previsoes, real
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"{newpath}/model_{run}.pth")
            paciencia = 30
        else:
            paciencia -= 1
        
        if percentagem % 10 == 0:
            print(f"Training {nome} | Progress: {percentagem}%")

        if paciencia == 0: 
            break

    ##############################################################################
    #   best_prev = lista das classes previstas pelo modelo                      #
    #   best_real = lista das classes reais das imagens                          #
    #   best_epoch_num = O numero da epoch na qual teve o melhor resultado de f1 #
    #   historico_f1 = Lista com os valores de f1-macro de todas as epochs       #
    ##############################################################################
    return best_prev, best_real, best_epoch_num, historico_f1

def treino_YOLO(path, nome, model, optimizer, run, ln, val_cos_lr, transform):
    seed_val = SEED[run]
    cos_lr = True if val_cos_lr == "CosineAnnealingLR" else False
    auto_aug = "autoaugment" if transform == "Autoaugmentation" else None   

    results = model.train(
            project = path,
            name = nome,
            data = "dados_split2", 
            imgsz = 224, 
            batch = 16,
            epochs = 200, 
            patience = 30, 
            seed = seed_val,
            optimizer = optimizer,
            lr0 = ln,
            cos_lr = cos_lr,
            auto_augment = auto_aug
            )

    preds = model.predict(source="dados_split2/test/*/*", save=False)

    y_pred = []
    y_true = []

    # mapa: nome da classe -> índice
    name_to_idx = {v: k for k, v in model.names.items()}

    for p in preds:
        y_pred.append(p.probs.top1)
        # extrair classe real (nome da pasta)
        class_name = os.path.basename(os.path.dirname(p.path))
        y_true.append(name_to_idx[class_name])

    return y_pred, y_true, None, None



def metricas(path, nome, run, best_prev, best_real, transform, df_resultados, best_epoch_num, historico_f1):
    ###################################################################
    #   Funcao que calcula metricas de accuracy,precision, recall, f1 #
    #   Cria grafico com os valores de f1/epoch                       #
    #   Cria uma matriz de confusao                                   #
    #   Atualiza um dataframe com os valores das metricas calculadas  #
    ###################################################################
    #Metricas
    accuracy = accuracy_score(best_real, best_prev)
    precision = precision_score(best_real, best_prev, average="macro")
    recall = recall_score(best_real, best_prev, average="macro") 
    f1 = f1_score(best_real, best_prev, average="macro") 

    if best_epoch_num:
        #grafico f1
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(historico_f1) + 1), historico_f1, marker='o', label='F1-Score (Macro)')
        plt.axvline(x=best_epoch_num, color='r', linestyle='--', label=f'Melhor epoca ({best_epoch_num})')

        plt.title(f'Evolucao do F1-Score - {nome}_{run}')
        plt.xlabel('epoca')
        plt.ylabel('F1-Score')
        plt.legend()
        plt.grid(True)

        # Salvar o grafico
        caminho_grafico = os.path.join(path, f"f1_evolution_{run}.png")
        plt.savefig(caminho_grafico, bbox_inches="tight")
        plt.close()
    
    #Matriz
    test_data = datasets.ImageFolder('dados_split2/test', transform=transform)
    classes = test_data.classes
    matriz = confusion_matrix(best_real, best_prev)
    display = ConfusionMatrixDisplay(matriz, display_labels=classes)
    display.plot(xticks_rotation=45)
    plt.xticks(rotation=45, ha='right')
    #Salvar a matriz
    caminho_imagem = os.path.join(path, f"confusion_matrix_{run}.png")
    plt.savefig(caminho_imagem, bbox_inches="tight")
    plt.close()
    
    #DataFrame
    nova_linha = {"nome": nome, "f1_macro": f1, "accuracy": accuracy, "precision_macro": precision, "recall_macro": recall, "best_epoch": best_epoch_num}
    df_resultados.loc[len(df_resultados)] = nova_linha

    return df_resultados


def modelos(model_name, learning_rate, optim = "AdamW"):
    ###################################################################
    #   Funcao que cria um modelo que sera posteriormente treinado    #
    ###################################################################
    if model_name == "resnet18":
        model = resnet18(ResNet18_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate)
    
    elif model_name == "resnet50":
        model = resnet50(ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate)

    elif model_name == "vit":
        model = vit_b_32(ViT_B_32_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=learning_rate)
    
    elif model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(MobileNet_V3_Small_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=learning_rate)
    
    elif model_name == "googlenet":
        model = googlenet(GoogLeNet_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate)
    
    elif model_name == "dino_vit":
        MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        class DinoV3Classifier(nn.Module):
            def __init__(self, model_id, num_classes):
                super().__init__()
                
                self.backbone = AutoModel.from_pretrained(model_id)
                
                for param in self.backbone.parameters():
                    param.requires_grad = False
                
                self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
            
            def forward(self, x):
                outputs = self.backbone(x)
                pooled = outputs.pooler_output
                logits = self.classifier(pooled)
                return logits

        model = DinoV3Classifier(MODEL_ID, num_classes)
        model.to(device)
        #optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    
    elif model_name == "vgg11":
        model = vgg11(VGG11_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False 
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=learning_rate)

    elif model_name == "convnext_tiny":
        model = convnext_tiny(ConvNeXt_Tiny_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=learning_rate)

    elif model_name == "convnext_small":
        model = convnext_small(ConvNeXt_Small_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=learning_rate)

    elif model_name == "convnext_base":
        model = convnext_base(ConvNeXt_Base_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=learning_rate)

    elif model_name == "efficientnet_v2_s":
        model = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=learning_rate)

    elif model_name == "efficientnet_v2_m":
        model = efficientnet_v2_m(EfficientNet_V2_M_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier[-1].parameters(), lr=learning_rate)

    elif model_name == "inception_v3":
        model = inception_v3(Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate)

    elif model_name == "densenet121":
        model = densenet121(DenseNet121_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        #optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    
    elif model_name == "se_resnext50":
        model = timm.create_model("seresnext50_32x4d", pretrained=True, num_classes=num_classes)

        for param in model.parameters():
            param.requires_grad = False

        classifier = model.get_classifier()
        for param in classifier.parameters():
            param.requires_grad = True

        #optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)


    elif model_name == "nasnet_large":
        model = timm.create_model('nasnetalarge', pretrained=True, num_classes=num_classes)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.get_classifier().parameters():
            param.requires_grad = True

        #optimizer = torch.optim.AdamW(model.get_classifier().parameters(),lr=learning_rate)
    
    elif model_name == "YOLOm":
        model = YOLO("YOLO_model/yolo26m-cls.pt")
        optimizer = None
        return model.to(device), optimizer

    elif model_name == "YOLOn":
        model = YOLO("YOLO_model/yolo26n-cls.pt")
        optimizer = None
        return model.to(device), optimizer
    
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    

    params = get_head_params(model, model_name)

    if optim == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    elif optim == "Adan":
        optimizer = Adan(params, lr = learning_rate, betas = (0.02, 0.08, 0.01), weight_decay = 0.02)                
                         
    # learning rate (can be much higher than Adam, up to 5-10x)
    # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
    # weight decay 0.02 is optimal per author


    return model.to(device), optimizer


def get_head_params(model, model_name):
    ###################################################################################
    #   Funcao que dependendo do modelo indicado, retorna a camada final do modelo    #
    ###################################################################################
    if model_name in ["resnet18", "resnet50", "googlenet", "inception_v3"]:
        return model.fc.parameters()
    
    elif model_name in ["vit"]:
        return model.heads.head.parameters()
    
    elif model_name in ["mobilenet_v3_small", "convnext_tiny", "convnext_small", "convnext_base","efficientnet_v2_s", "efficientnet_v2_m", "vgg11"]:
        return model.classifier[-1].parameters()
    
    elif model_name == "dino_vit":
        return model.classifier.parameters()
    
    elif model_name == "densenet121":
        return model.classifier.parameters()
    
    elif model_name in ["se_resnext50", "nasnet_large"]:
        return model.get_classifier().parameters()
    
    else:
        raise ValueError(f"Modelo desconhecido para get_head_params: {model_name}")


def train_and_evaluate_models(path, file_path, modelos_config, run, df_resultados):
    #########################################################################################
    #   Funcao que concatena todas as funcoes a cima em uma so funcao                       #
    #   Esta funcao cria a arquitetura, treina a mesma e depois calcula suas metricas       #
    #   Cria um CSV com as metricas do modelo treinado e os modelos anteriormente treinados #
    #########################################################################################
    
    for name, config in modelos_config.items():
        newpath = f"{path}/{name}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        test_loader, train_loader = img_process(config["train_transform"], config["test_transform"])

        model, optimizer = modelos(name, config["learning_rate"], config["optimizer"])

        if name in ["YOLOm", "YOLOn"]:
            transform = None
            if config["train_transform"] == training_transforms_6 or config["train_transform"]=="augmentation_6":
                transform = "Autoaugmentation"
            elif ("augmentation" in newpath and config["train_transform"] != training_transforms_6) or config["scheduler"] in ["StepLR", "ReduceLROnPlateau"]:
                return df_resultados
                
            best_prev, best_real, best_epoch_num, historico_f1 = treino_YOLO(newpath, name, model, config["optimizer"], run, config["learning_rate"], config["scheduler"], transform)
        
        else:
            best_prev, best_real, best_epoch_num, historico_f1 = best_epoch(newpath, name, run, model, optimizer, config["scheduler"], config["loss_type"], train_loader, test_loader)
        

        df_resultados = metricas(newpath, name, run, best_prev, best_real, config["test_transform"], df_resultados, best_epoch_num, historico_f1)

        df_resultados.to_csv(file_path, index=False)
    
    return df_resultados


def load_or_create(file_path):
    ###################################################################################
    #   Funcao que retorna um arquivo CSV, se nao houver, cria e retorna um dataframe #
    ###################################################################################
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["nome", "f1_macro", "accuracy", "precision_macro", "recall_macro", "best_epoch"])


def run_experiments(path, file_path, config, n_runs=3):
    ###########################################################
    #   Funcao que define uma seed e depois treina os modelos #
    ###########################################################
    df = load_or_create(file_path)
    if n_runs > 5:
        print("Valores a cima de 5 nao sao suportados")
        n_runs = 5

    for run in range(n_runs):
        torch.manual_seed(SEED[run])
        train_and_evaluate_models(path, file_path, config, run, df)

    return df


def read_results(variavel_tipo):
    path = "teste"
    tables = []
    if variavel_tipo == "augmentation":
        dados_controlo = pd.read_csv(f"{path}/dados_controlo/resultados_modelos_iniciais.csv")
        dados_controlo["nome_var"] = "controlo"
        tables.append(dados_controlo)

    for subfolder in os.listdir(path):
        if variavel_tipo not in subfolder:
            continue

        subfolder_path = os.path.join(path, subfolder)

        for csv_file in glob.glob(os.path.join(subfolder_path, "*.csv")):
            table = pd.read_csv(csv_file)
            table["nome_var"] = subfolder
            tables.append(table)

    new_table = pd.concat(tables, ignore_index=True).sort_values(by="f1_macro", ascending=False).drop_duplicates(subset="nome", keep="first").set_index("nome")
    return new_table


def change_dictionary(variavel_tipo):

    new_table = read_results(variavel_tipo)

    for model_name, row in new_table.iterrows():
        nome_var = row["nome_var"]

        try:
            if variavel_tipo == "augmentation":
                if "YOLO" in model_name and nome_var == "augmentation_6":
                    modelos_config[model_name]["train_transform"] = nome_var
                else:
                    modelos_config[model_name]["train_transform"] = TRANSFORMS[nome_var]

                
            if variavel_tipo == "optimizer":
                change_dictionary(variavel_tipo="augmentation")
                parts = nome_var.split("_")
                if parts[0] == "controlo" and "Yolo" not in model_name:
                    modelos_config[model_name]["optimizer"] = "Adamw"
                    modelos_config[model_name]["learning_rate"] = 0.001

                elif parts[0] == "controlo" and "Yolo" in model_name:
                    modelos_config[model_name]["optimizer"] = "auto"
                    modelos_config[model_name]["learning_rate"] = 0.001

                else:
                    modelos_config[model_name]["optimizer"] = parts[1]
                    modelos_config[model_name]["learning_rate"] = float(parts[2])

            if variavel_tipo == "scheduler":
                change_dictionary(variavel_tipo="optimizer")
                parts = nome_var.split("_")
                if parts[0] == "controlo":
                    modelos_config[model_name]["scheduler"] = None
                else:
                    modelos_config[model_name]["scheduler"] = parts[1]

        except:
            pass


def output_images(classe):
    
    tipo = input("A imagm é de um membro Esquerdo (E) ou Direito (D)?").upper()

    while tipo not in ["D", "E"]:
        tipo = input("Invalido! A imagm é de um membro Esquerdo (E) ou Direito (D)?").upper()

    imagens_classes = []
    pasta=f"imagens_classes\{classe[:3]}_{tipo}_{classe[4:]}"
    for imagem in os.listdir(pasta):
        imagens_classes.append(os.path.join(pasta, imagem))

    fig, axs = plt.subplots(1, len(imagens_classes), figsize=(12, 4))

    if len(imagens_classes) == 1:
        axs = [axs]

    for ax, caminho in zip(axs, imagens_classes):
        img = Image.open(caminho)
        ax.imshow(img)
        ax.axis("off")

    plt.show()


def classify_image(image_path):
    caminho_do_melhor_modelo = "runs/classify/teste/scheduler_CosineAnnealingLR/YOLOm/YOLOm/weights/best.pt" 
    model = YOLO(caminho_do_melhor_modelo)

    preds = model.predict(source=image_path)

    for p in preds:
        # pegamos na classe com maior probabilidade
        top1_idx = p.probs.top1
        class_name = model.names[top1_idx]
        confidence = p.probs.top1conf.item() * 100
        if confidence <= 75:
            return "Imagem inválida"
        else:
            print(f"Classe: {class_name}")
            output_images(class_name)
            