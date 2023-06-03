#METODOLOGÍAS NO SUPERVISADAS

#PREPARACIÓN DE DATA
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
#Visualización del diagrama de caja de una de las variables limpiadas, la cual acota su rango entre 0-40 a 0-0.3 luego de eliminar outlayers
boxplot(cleaned_data$word_freq_3d)
boxplot(data_mal$V4)

data <- cleaned_data
#Importar librerías
library(DataExplorer)
library(factoextra)
library(caret)
library(plotly)
library(readr)
library(pROC)
library(MASS)
library(dplyr)
library(ggpubr)
library(ggplot2)
library(tidyverse)
library(caret)
library(e1071)
library(caTools)

plot_intro(data)

plot_bar(data)

plot_histogram(data)

#ELIMINACIÓN DE VARIABLE NOMINAL"SPAM"
features <- data[, 1:57]
spam <- data[, 58]

view (spam)
view(features)

#CORRELACIÓN ENTRE LOS ATRIBUTOS
cor(features)

plot_correlation(features)


#PCA estandarización 


apply(X = features, MARGIN = 2, FUN = mean)

apply(X = features, MARGIN = 2, FUN = var)

#DEFINICIÓN DE MODELO PCA

pca <- prcomp(features, scale = TRUE)
names(pca)

#Varianza explicada por componente 

prop_varianza <- pca$sdev^2 / sum(pca$sdev^2)

pca_var<-ggplot(data = data.frame(prop_varianza, pc = 1:length(prop_varianza)), aes(x = pc, y = prop_varianza)) + 
  geom_col(width = 0.3) +  scale_y_continuous(limits = c(0,1)) +  theme_bw() +
  labs(x = "Componente principal", y = "Prop. de varianza explicada")
pca_var

#Varianza explicada de las primeras componentes 

fviz_eig(pca)

fviz_pca_biplot(pca, repel = TRUE,
                col.var = "#D2691E", # color para las variables
                col.ind = "#000000"  # colores de las estaciones
)


# Varianza explicada acumulada

prop_varianza_acum <- cumsum(prop_varianza)

pca_var_acum<-ggplot(data = data.frame(prop_varianza_acum, pc = 1:length(prop_varianza)), aes(x = pc, y = prop_varianza_acum, group = 1)) +
  geom_point() +  geom_line() +  theme_bw() +  labs(x = "Componente principal", y = "Prop. varianza explicada acumulada")

pca_var_acum

biplot(pca)

# Gráfico de las dos PCA 1 y PCA 2
plot_prcomp(data)
biplot(x = pca, scale = 0, cex = 0.6, col = c("blue4", "brown3"))


# Importancia de cada variable

library(corrplot)
var <- get_pca_var(pca)
corrplot(var$cos2, is.corr = FALSE)

#K-MEANS

#Semilla de reproducibilidad 

set.seed(411)
#Definición de modelos de K-Means para diferentes números de clústeres
k2 <- kmeans(features, centers = 2, nstart = 25)
k3 <- kmeans(features, centers = 3, nstart = 25)
k4 <- kmeans(features, centers = 4, nstart = 25)
k5 <- kmeans(features, centers = 5, nstart = 25)
#Definición de gráficos de los clústeres 
p2 <- fviz_cluster(k2, geom = "point", data = features) + ggtitle("k = 2")
p3 <- fviz_cluster(k3, geom = "point", data = features) + ggtitle("k = 3")
p4 <- fviz_cluster(k4, geom = "point", data = features) + ggtitle("k = 4")
p5 <- fviz_cluster(k5, geom = "point", data = features) + ggtitle("k = 5")
#Muestra los gráficos ya definidos
print(p2)
print(p3)
print(p4)
print(p5)