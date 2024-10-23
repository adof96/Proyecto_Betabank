# Proyecto_Betabank
Los clientes de Beta Bank se están yendo, cada mes, poco a poco. Los banqueros descubrieron que es más barato salvar a los clientes existentes que atraer nuevos.

Necesitamos predecir si un cliente dejará el banco pronto. Tenemos los datos sobre el comportamiento pasado de los clientes y la terminación de contratos con el banco.

Crearemos un modelo con el máximo valor F1 posible. Necesitamos un valor F1 de al menos 0.59. Verificaremos F1 para el conjunto de prueba. 
Además, debemos medir la métrica AUC-ROC y compararla con el valor F1

## Herramientas utilizadas
- python
- pandas
- matplotlib
- scikit-learn

## Pasos

1.  Inicializacion y carga de datos
2.  Estandarizacion de datos
2.1.  Codificacion OHE y escalado de caracteristicas
3.  Examinar el equilibrio de clases
3.1.  Prueba de consistencia
3.2.  Evaluacion del modelo
4.  Mejora de la calidad del modelo
4.1.  Sobremuestreo y submuestreo
4.2.  Regresion logistica
4.3.  Bosque aleatorio
5.  Curva Roc