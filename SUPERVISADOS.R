dataframe <- data.frame(spambase)

colnames(dataframe) <- c("word_freq_make",         
                    "word_freq_address",      
                    "word_freq_all",          
                    "word_freq_3d",           
                    "word_freq_our",          
                    "word_freq_over",         
                    "word_freq_remove",       
                    "word_freq_internet",     
                    "word_freq_order",        
                    "word_freq_mail",         
                    "word_freq_receive",      
                    "word_freq_will",         
                    "word_freq_people",       
                    "word_freq_report",       
                    "word_freq_addresses",    
                    "word_freq_free",         
                    "word_freq_business",     
                    "word_freq_email",        
                    "word_freq_you",          
                    "word_freq_credit",       
                    "word_freq_your",         
                    "word_freq_font",         
                    "word_freq_000",          
                    "word_freq_money",        
                    "word_freq_hp",           
                    "word_freq_hpl",          
                    "word_freq_george",       
                    "word_freq_650",          
                    "word_freq_lab",          
                    "word_freq_labs",         
                    "word_freq_telnet",       
                    "word_freq_857",          
                    "word_freq_data",         
                    "word_freq_415",          
                    "word_freq_85",           
                    "word_freq_technology",   
                    "word_freq_1999",         
                    "word_freq_parts",        
                    "word_freq_pm",           
                    "word_freq_direct",       
                    "word_freq_cs",           
                    "word_freq_meeting",      
                    "word_freq_original",     
                    "word_freq_project",      
                    "word_freq_re",           
                    "word_freq_edu",          
                    "word_freq_table",        
                    "word_freq_conference",   
                    "char_freq_punto_coma",            
                    "char_freq_left_par_red",            
                    "char_freq_left_par_cua",            
                    "char_freq_exclamacion",            
                    "char_freq_peso",            
                    "char_freq_gato",            
                    "capital_run_length_average", 
                    "capital_run_length_longest", 
                    "capital_run_length_total",
                    "spam")

# LIMPIEZA DE DATOS

View(dataframe)

# Seleccionar solo las columnas numéricas
numeric_columns <- sapply(dataframe, is.numeric)

# Calcular el coeficiente de variación para cada columna
cv <- apply(dataframe[, numeric_columns], 2, function(x) sd(x) / mean(x) * 100)

# Calcular el promedio y la desviación estándar de cada columna
mean_values <- apply(dataframe[, numeric_columns], 2, mean)
sd_values <- apply(dataframe[, numeric_columns], 2, sd)

# Identificar las columnas con coeficiente de variación mayor al promedio
columns_to_clean <- names(cv[cv > mean(cv) + sd(cv)])

# Realizar la limpieza eliminando las filas fuera del rango para las columnas seleccionadas
cleaned_data <- dataframe[!(apply(dataframe[, columns_to_clean], 1, function(x) any(x < mean_values - 3 * sd_values | x > mean_values + 3 * sd_values))), ]

# Imprimir el dataframe limpio
View(cleaned_data)

data_mal <- spambase

boxplot(cleaned_data$word_freq_3d)
boxplot(data_mal$V4)

data <- cleaned_data

library(tidyverse)
library(caret)
library(ggplot2)
library(e1071)
library(DataExplorer)
library(caTools)
library(plotly)
library(readr)
library(pROC)
library(MASS)
library(dplyr)

# ANÁLISIS EXPLORATORIO DE DATOS

plot_intro(data)  # Resumen general del conjunto de datos
plot_missing(data)  # Gráfico de barras que muestra los valores perdidos en cada columna
plot_histogram(data)  # Histogramas para las variables numéricas
plot_correlation(data)  # Matriz de correlación entre las variables numéricas
plot_boxplot(data)  # Diagramas de caja para las variables numéricas


# MÉTODOS SUPERVISADOS

set.seed(123)  # Establecer semilla para reproducibilidad
train_idx <- sample(nrow(data), nrow(data) * 0.7)  # Índices para el conjunto de entrenamiento (70% de los datos)
train <- data[train_idx, ]  # Datos de entrenamiento
test <- data[-train_idx, ]  

# Regresión Logística
model_RL <- glm(spam ~ ., data = train, family = "binomial")
summary(model_RL)

# Predicciones
predictions_RL <- predict(model_RL, newdata = test, type = "response")

y_pred_RL <- ifelse(predictions_RL > 0.6, 1, 0)
y_pred_RL <- as.factor(y_pred_RL)

# Curva ROC
roc_RL <- roc(test$spam, predictions_RL)
plot(roc_RL, main = "Curva ROC - Regresión Logística")

# Área bajo la curva ROC
AUC_RL <- auc(roc_RL)

# Matriz de confusión
test$spam <- as.factor(test$spam)
confusion_matrix_RL <- confusionMatrix(data = y_pred_RL, reference = test$spam)
confusion_matrix_RL



# LDA
model_LDA <- lda(spam ~ ., data = train)
summary(model_LDA)

# Predicciones
predictions_LDA <- predict(model_LDA, newdata = test)$posterior[, 2]

y_pred_LDA <- ifelse(predictions_LDA > 0.6, 1, 0)
y_pred_LDA <- as.factor(y_pred_LDA)

# Curva ROC
roc_LDA <- roc(test$spam, predictions_LDA)
plot(roc_LDA, main = "Curva ROC - LDA")

# Área bajo la curva ROC
AUC_LDA <- auc(roc_LDA)

# Matriz de confusión
confusion_matrix_LDA <- confusionMatrix(data = y_pred_LDA, reference = test$spam)
confusion_matrix_LDA


library(caret)

# QDA
model_QDA <- qda(spam ~ ., data = train)
summary(model_QDA)

# Predicciones
predictions_QDA <- predict(model_QDA, newdata = test)$posterior[, 2]

y_pred_QDA <- ifelse(predictions_QDA > 0.6, 1, 0)
y_pred_QDA <- as.factor(y_pred_QDA)

# Curva ROC
roc_QDA <- roc(test$spam, predictions_QDA)
plot(roc_QDA, main = "Curva ROC - QDA")

# Área bajo la curva ROC
AUC_QDA <- auc(roc_QDA)

# Matriz de confusión
confusion_matrix_QDA <- confusionMatrix(data = y_pred_QDA, reference = test$spam)
confusion_matrix_QDA





library(dplyr)
library(class)
library(caret)

set.seed(163)

# Paso 1: Separar los conjuntos de entrenamiento y prueba (train y test ya deben estar definidos)
train_indices <- sample(1:nrow(data), nrow(data) * 0.7) 
train_knn <- data[train_indices, ]
test_knn <- data[-train_indices, ]

# Paso 2: Preparar los datos de entrenamiento y prueba
train_knn <- train_knn %>% dplyr::select(-spam)
test_knn <- test_knn %>% dplyr::select(-spam)

# Paso 3: Entrenar y evaluar el modelo KNN
k <- 5
knn_pred <- knn(train_knn, test_knn, train$spam, k = k, prob = TRUE)

# Paso 4: Evaluar el rendimiento del modelo
confusion_matrix <- table(knn_pred, test$spam)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Paso 5: Mostrar los resultados
cat("Precisión del modelo KNN:", accuracy, "\n")
confusion_matrix_knn <- confusionMatrix(data = knn_pred, reference = test$spam)
confusion_matrix_knn
