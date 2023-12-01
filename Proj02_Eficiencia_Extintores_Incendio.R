setwd("E:/DataScience/FCD/BigDataRAzure/Cap20_Projetos_com_FeedBack/Proj02_PrevEficExtintoresIncendio")     

#===========Problema de Negócio=========

#Seria possível usar Machine Learning para prever o funcionamento de um extintor de
#incêndio com base em simulações feitas em computador e assim incluir uma camada adicional
#de segurança nas operações de uma empresa? 
#Usando dados reais disponíveis publicamente, seu trabalho é desenvolver um modelo
#de Machine Learning capaz de prever a eficiência de extintores de incêndio.

# Explicação:
# SHORT DESCRIPTION: The dataset was obtained as a result of the 
# extinguishing tests of four different fuel flames with a sound 
# wave extinguishing system. The sound wave fire-extinguishing 
# system consists of 4 subwoofers with a total power of 4,000 
# Watt placed in the collimator cabinet. There are two amplifiers 
# that enable the sound come to these subwoofers as boosted. 
# Power supply that powers the system and filter circuit ensuring 
# that the sound frequencies are properly transmitted to the 
# system is located within the control unit. While computer 
# is used as frequency source, anemometer was used to measure 
# the airflow resulted from sound waves during the extinguishing 
# phase of the flame, and a decibel meter to measure the sound 
# intensity. An infrared thermometer was used to measure the 
# temperature of the flame and the fuel can, and a camera is 
# installed to detect the extinction time of the flame. A total 
# of 17,442 tests were conducted with this experimental setup. 

# The experiments are planned as follows:

# FUEL - Three different liquid fuels and LPG fuel were used to 
# create the flame.

# SIZE - 5 different sizes of liquid fuel cans are used to achieve 
# different size of flames. Half and full gas adjustment is used for LPG fuel.

# DISTANCE - While carrying out each experiment, the fuel container, at 
# 10 cm distance, was moved forward up to 190 cm by increasing 
# the distance by 10 cm each time.

# DESIBEL AND AIRFLOW - Along with the fuel container, anemometer and decibel meter 
# were moved forward in the same dimensions.

# FREQUENCY -  Fire extinguishing experiments was conducted with 54 
# different frequency sound waves at each distance and flame size.
# 
# Throughout the flame extinguishing experiments, the data obtained from each 
# measurement device was recorded and a dataset was created. The
# dataset includes the features of fuel container size representing the 
# flame size, fuel type, frequency, decibel, distance, airflow and flame 
# extinction. Accordingly, 6 input features and 1 output feature 
# will be used in models. The explanation of a total of seven 
# features for liquid fuels in the dataset is given in Table 1, 
# and the explanation of 7 features for LPG fuel is given in Table 2.
# The status property (flame extinction or non-extinction states) 
# can be predicted by using six features in the dataset. Status 
# and fuel features are categorical, while other features are 
# numerical. 8,759 of the 17,442 test results are the 
# non-extinguishing state of the flame. 8,683 of them are the 
# extinction state of the flame. According to these numbers, it 
# can be said that the class distribution of the dataset is 
# almost equal.

library(readxl)
library(dplyr)
library(caret)
#==========Carregando os dados==========

dados <- read_excel("Acoustic_Extinguisher_Fire_Dataset.xlsx", 
                    sheet = "A_E_Fire_Dataset")

#========Resumo dos Dados===============:
View(dados)

dim(dados)

str(dados)

#=========Explorando os dados ==============

# Resumo de variáveis :


summary(dados$DISTANCE)
summary(dados$DESIBEL)


plot(x = dados$AIRFLOW, y = dados$DESIBEL,
     main = " ScatterPlot - AIRFLOW x Decibel",
     xlab = "AIRFLOW",
     ylab = " Decibel")

library(ggplot2)

ggplot(dados, aes(x = FREQUENCY, y  = AIRFLOW)) +
  geom_point() +
  facet_wrap(~ STATUS)


ggplot(dados, aes(x = STATUS)) +
  geom_bar() +
  facet_wrap(~ FUEL)


ggplot(dados, aes(x = DISTANCE)) +
  geom_bar() +
  facet_wrap(~ STATUS)


table(dados$FUEL)
table(dados$SIZE)

library(gmodels)

CrossTable(x = dados$FUEL, y = dados$STATUS)

CrossTable(x = dados$FUEL, y = dados$STATUS, chisq = TRUE)

# transformando variáveis em fator:

# obs : Na minha avaliação os números da coluna size são classificações que 
# representam os valores do tamanho do recipiente, assim, também 
# transformei esta variável para o tipo fator:

dados$SIZE <- as.factor(dados$SIZE)
dados$FUEL <- as.factor(dados$FUEL)
dados$STATUS <- as.factor(dados$STATUS)

str(dados)

# Verificando valores ausentes

sum(is.na(dados))
complete_cases <- sum(complete.cases(dados))
complete_cases


# Renomeando as colunas:

dados <- as.data.frame(dados)
str(dados)
dados <- rename(dados, size = SIZE, fuel = FUEL, distance = DISTANCE, 
                desibel = DESIBEL, airflow = AIRFLOW, 
                frequency = FREQUENCY, status  = STATUS)
View(dados)




# Normalizando os dados:
library(scales)

dados_norm <- as.data.frame(lapply(dados[3:6], rescale))

dados_norm <- cbind(dados_norm, dados[,c(1:2,7)])

dados_norm$status <- sapply(dados_norm$status, function(x){
  ifelse(x == "1", "extinto", "nao_extinto" )
})

dados_norm$status <- factor(dados_norm$status, 
                            levels = c("extinto", "nao_extinto"),
                            labels = c("extinto", "nao_extinto"))

View(dados_norm)
str(dados_norm)

#===========Criação do primeiro modelo com o algoritimo Random forest=====

# Criando dados de teste e treino:

splitData <- function(dataframe, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)/2))# trunc arredonda 
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset = trainset, testset = testset)
}

# Gerando dados de treino e de teste

splits <- splitData(dados_norm, seed = 808)

# Separando os dados:

dados_treino <- splits$trainset
dados_teste <- splits$testset
View(dados_treino)
# Verificando o numero de linhas
nrow(dados_treino)
nrow(dados_teste)

# Verificando as varáveis mais importantes:

library(randomForest)

modelo <- randomForest(status ~ .,
                       data = dados_treino,
                       ntree = 100,
                       nodesize = 10, 
                       importance = TRUE)

varImpPlot(modelo)


# Obs: A variável decibel foi excluída após análise da visualização do 
# teste de importância:

modelo <- randomForest(status ~ distance 
                       + airflow  
                       + size 
                       + frequency 
                       + fuel,
                       data = dados_treino,
                       ntree = 100,
                       nodesize = 10)
print(modelo)

# Previsão e Confusion MAtrix com dados de treino:

previsto <- predict(modelo, dados_treino)

confusionMatrix(previsto, dados_treino$status)

# A acuracia do modelo foi igual a 98% com dados de treino

# Previsão e confusion Matrix com dados de teste:

previsto2 <- predict(modelo, dados_teste)

confusionMatrix(previsto2, dados_teste$status)

# A Acurácia do modelo foi igual de 96% com dados de teste


# Visualizando o resultado

plot(modelo)



# Gerando as classes de dados

library("ROCR")

class1 <- predict(modelo, newdata = dados_teste, type = 'prob')

head(class1)

class2 <- dados_teste$status

head(class2)

pred <- prediction(class1[,2], class2)
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = rainbow(10))

# Obs : Por meio da característica da curva ROC, ao se apresentara bem 
# próxima do canto superior esquerdo, podemos afirmar que o modelo apresenta
# um excelente desempenho como classificador.

# ======Criação de um segundo modelo alterando os dados de treino e teste=======

set.seed(222)
ind <- sample(2, nrow(dados_norm), replace = TRUE, prob = c(0.7, 0.3))
train <- dados_norm[ind==1,]
test <- dados_norm[ind==2,]


# Verificando a importância das variáveis preditoras com o random forest

modelo2 <- randomForest(status ~ .,
                        data = train,
                        ntree = 100,
                        nodesize = 10, 
                        importance = TRUE)

# Visualizando  a Importância:

varImpPlot(modelo)

# Construindo o modelo:

modelo2 <- randomForest(status ~ .,
                        data = train,
                        ntree = 100,
                        nodesize = 10)
print(modelo2)

# Previsão e Confusion MAtrix com dados de treino:

previsto3 <- predict(modelo2, train)

confusionMatrix(previsto3, train$status)

# A acuracia do modelo 2 é igual a 98% com dados de treino

# Previsão e confusion Matrix com dados de teste:

previsto4 <- predict(modelo2, test)

confusionMatrix(previsto4, test$status)

# A acurácia do modelo 2 é igual a 96% com dados de teste


# Visualizando o resultado

plot(modelo2)

# Gerando as classes de dados


class3 <- predict(modelo2, newdata = test, type = 'prob')

head(class3)

class4 <- test$status

head(class2)

pred <- prediction(class3[,2], class4)
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = rainbow(10))

# Obs : Por meio da característica da curva ROC, ao se apresentara bem 
# próxima do canto superior esquerdo, podemos afirmar que o modelo2 apresenta
# um excelente desempenho como classificador.

#===============================

# Baseado na acurácia do modelo com dados de treino e teste,
# foi escolhido o modelo 2 como capaz de prever a eficiência dos extintores.




