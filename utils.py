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
    test_data = datasets.ImageFolder('dados_split/test', transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    train_data = datasets.ImageFolder('dados_split/train', transform=train_transforms)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    return test_loader, train_loader


###################################
#   Funcoes de treino             #
###################################
def best_epoch(path, nome, model, optimizer, loss_type, train_loader, test_loader, num_epochs):
    #####################################################################################
    #   Funco que treina e testa um modelo a cada epoch para achar o melhor valor de f1 #
    #####################################################################################
    
    best_f1 = -1
    best_epoch_num = 0
    historico_f1 = []

    #Utilizar CPU ou cuda
    model.to(device)

    if loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()

    elif loss_type == "kl_div":
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
    

    #Treinar e avaliar o modelo
    for epoch in range(1, num_epochs +1):
        model.train()
        running_loss = 0.0
        percentagem = int((epoch/num_epochs) * 100)

        #Treinar o modelo
        for imagens, labels in train_loader:
            imagens = imagens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imagens)


            if loss_type == "cross_entropy":
                loss = criterion(outputs, labels)

            elif loss_type == "kl_div":
                log_probs = F.log_softmax(outputs, dim=1)
                target = F.one_hot(labels, num_classes=outputs.shape[1]).float()
                loss = criterion(log_probs, target)


            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        
        #Avaliar o modelo
        model.eval()
        previsoes = []
        real = []

        with torch.no_grad():
            for imagens, labels in test_loader:
                imagens, labels = imagens.to(device), labels.to(device)
                outputs = model(imagens)
                
                _, predicted = torch.max(outputs, 1) 
                
                previsoes.extend(predicted.cpu().numpy()) 
                real.extend(labels.cpu().numpy())
        

        #calcular metrica
        f1 = f1_score(real, previsoes, average="macro") 
        historico_f1.append(f1)
            
        if f1 >= best_f1:
            best_f1 = f1
            best_epoch_num = epoch
            best_prev, best_real = previsoes, real
            torch.save(model.state_dict(), f"{path}/model_{nome}.pth")
        

        if percentagem % 10 == 0:
            print(f"Training {nome} | Progress: {percentagem}%")

    ##############################################################################
    #   best_prev = lista das classes previstas pelo modelo                      #
    #   best_real = lista das classes reais das imagens                          #
    #   best_epoch_num = O numero da epoch na qual teve o melhor resultado de f1 #
    #   historico_f1 = Lista com os valores de f1-macro de todas as epochs       #
    ##############################################################################
    return best_prev, best_real, best_epoch_num, historico_f1

def metricas(path, nome, best_prev, best_real, transform, df_resultados, best_epoch_num, historico_f1):
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

    #grafico f1
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(historico_f1) + 1), historico_f1, marker='o', label='F1-Score (Macro)')
    plt.axvline(x=best_epoch_num, color='r', linestyle='--', label=f'Melhor epoca ({best_epoch_num})')

    plt.title(f'Evolucao do F1-Score - {nome}')
    plt.xlabel('epoca')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)

    # Salvar o grafico
    caminho_grafico = os.path.join(path, f"{nome}_f1_evolution.png")
    plt.savefig(caminho_grafico, bbox_inches="tight")
    plt.close()
    
    #Matriz
    test_data = datasets.ImageFolder('dados_split/test', transform=transform)
    classes = test_data.classes
    matriz = confusion_matrix(best_real, best_prev)
    display = ConfusionMatrixDisplay(matriz, display_labels=classes)
    display.plot(xticks_rotation=45)
    plt.xticks(rotation=45, ha='right')
    #Salvar a matriz
    caminho_imagem = os.path.join(path, f"{nome}_confusion_matrix.png")
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
    
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    

    params = get_head_params(model, model_name)

    if optim == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)

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

def train_and_evaluate_models(path, file_path, modelos_config, train_transforms, test_transforms, df_resultados):
    #########################################################################################
    #   Funcao que concatena todas as funcoes a cima em uma so funcao                       #
    #   Esta funcao cria a arquitetura, treina a mesma e depois calcula suas metricas       #
    #   Cria um CSV com as metricas do modelo treinado e os modelos anteriormente treinados #
    #########################################################################################
    test_loader, train_loader = img_process(train_transforms, test_transforms)

    for name, config in modelos_config.items():
        model, optimizer = modelos(name, config["learning_rate"], config["optimizer"])
        best_prev, best_real, best_epoch_num, historico_f1 = best_epoch(path, name, model, optimizer, config["loss_type"], train_loader, test_loader, config["num_epochs"])
        df_resultados = metricas(path, name, best_prev, best_real, test_transforms, df_resultados, best_epoch_num, historico_f1)

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

def run_experiments(file_path, config, train_tf, val_tf, path, n_runs=3):
    ###########################################################
    #   Funcao que define uma seed e depois treina os modelos #
    ###########################################################
    df = load_or_create(file_path)
    if n_runs > 5:
        print("Valores a cima de 5 nao sao suportados")
        n_runs = 5

    for n in range(n_runs):
        torch.manual_seed(seed[n])
        train_and_evaluate_models(path, file_path, config, train_tf, val_tf, df)

    return df
