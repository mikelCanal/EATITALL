
import sys
sys.path.insert(1, '/home/eatitall_scripts')
sys.path.insert(1, '/home/root/pctobs/lib/python3.8/site-packages')
import pandas as pd

df = pd.read_csv('./archivos/datos_crudos.csv')
df_ejemplo=df.head(10)

obs1="Test de cetonas en sangre. El paciente ha experimentado mejoras en el control glucémico bajo tratamiento con Metformina, lo cual se ha complementado con una dieta enfocada en la reducción de carbodratos refinados. La evitación de pan blanco y dulces, junto con la preferencia por carbohidratos complejos como la avena, quinoa y legumbres, son cruciales para potenciar los efectos beneficiosos de la Metformina y mantener una estabilidad en los niveles de glucosa en sangre."
obs2="Con el régimen de insulina basal-bolo, es vital el conteo preciso de carbohidratos para ajustar adecuadamente las dosis de insulina prandial. La alimentación debe enfocarse en evitar el consumo irregular de carbohidratos y planificar comidas equilibradas, reduciendo la ingesta de alimentos altos en azúcares simples para evitar desequilibrios glucémicos."
obs3="La efectividad del tratamiento con inhibidores de SGLT2 se ve reforzada por mantener una hidratación adecuada, dada la tendencia a la deshidratación asociada con estos medicamentos. Se enfatiza la importancia de beber suficiente agua y limitar el consumo de bebidas azucaradas para apoyar el control glucémico y la salud renal."
obs4="El tratamiento con agonistas de GLP-1, que mejora el apetito y promueve la pérdida de peso, se optimiza con una dieta baja en grasas saturadas. Se recomienda incluir alimentos como el aceite de oliva, aguacates y frutos secos, que aportan grasas saludables y contribuyen a un mejor manejo metabólico."
obs5="Polifagia. Al estar en tratamiento con Dapagliflozina, se incrementa el riesgo de infecciones genitourinarias, lo cual se puede manejar con una dieta rica en líquidos y un consumo moderado de azúcares y carbohidratos. Esta estrategia dietética ayuda a minimizar el riesgo de infecciones y apoya la eficacia del medicamento."
obs6="La asociación del tratamiento con Pioglitazona con el aumento de peso resalta la necesidad de una dieta equilibrada y baja en calorías. Evitar alimentos procesados y bebidas azucaradas es fundamental para controlar el aumento de peso y mantener un perfil glucémico estable."
obs7="La estabilidad en los niveles de glucosa lograda con los inhibidores de la DPP-4 se complementa con una dieta rica en fibra y baja en carbohidratos simples. Incrementar la ingesta de vegetales, especialmente los de hoja verde, es beneficioso para el control glucémico y la salud general."
obs8="La necesidad de ajustar la dosis de insulina en días de actividad física intensa subraya la importancia de consumir snacks ricos en carbohidratos complejos antes del ejercicio. Esto ayuda a prevenir hipoglucemias y asegura un suministro energético adecuado durante la actividad física."
obs9="Con el tratamiento de Repaglinida, es esencial no omitir comidas para evitar hipoglucemias. Adoptar un plan de alimentación estructurado que incluya snacks saludables entre comidas asegura la estabilidad glucémica y apoya la eficacia del tratamiento."
obs10="El tratamiento con Sulfonylureas exige una vigilancia constante de los síntomas de hipoglucemia. Disponer de snacks rápidos y saludables ricos en glucosa, como fruta fresca o jugos sin azúcar añadido, es crucial para responder de manera efectiva a cualquier signo de hipoglucemia."

observaciones = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10]

# Si el DataFrame no tiene exactamente 10 filas, este paso fallará
if len(df_ejemplo) == 10:
    df_ejemplo['observaciones'] = observaciones
else:
    print("El DataFrame no tiene 10 filas. Tiene {} filas.".format(len(df_ejemplo)))

df_ejemplo.to_csv('./archivos/datos_con_10_ejemplos_v3.csv', index=False)