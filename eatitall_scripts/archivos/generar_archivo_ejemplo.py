
import sys
sys.path.insert(1, '/home/eatitall_scripts')
sys.path.insert(1, '/home/root/pctobs/lib/python3.8/site-packages')
import pandas as pd

df = pd.read_csv('./archivos/datos_crudos.csv')
df_ejemplo=df.head(10)

# obs1="Test de cetonas en sangre. El paciente ha experimentado mejoras en el control glucémico bajo tratamiento con Metformina, lo cual se ha complementado con una dieta enfocada en la reducción de carbodratos refinados. La evitación de pan blanco y dulces, junto con la preferencia por carbohidratos complejos como la avena, quinoa y legumbres, son cruciales para potenciar los efectos beneficiosos de la Metformina y mantener una estabilidad en los niveles de glucosa en sangre."
# obs2="Con el régimen de insulina basal-bolo, es vital el conteo preciso de carbohidratos para ajustar adecuadamente las dosis de insulina prandial. La alimentación debe enfocarse en evitar el consumo irregular de carbohidratos y planificar comidas equilibradas, reduciendo la ingesta de alimentos altos en azúcares simples para evitar desequilibrios glucémicos."
# obs3="La efectividad del tratamiento con inhibidores de SGLT2 se ve reforzada por mantener una hidratación adecuada, dada la tendencia a la deshidratación asociada con estos medicamentos. Se enfatiza la importancia de beber suficiente agua y limitar el consumo de bebidas azucaradas para apoyar el control glucémico y la salud renal."
# obs4="El tratamiento con agonistas de GLP-1, que mejora el apetito y promueve la pérdida de peso, se optimiza con una dieta baja en grasas saturadas. Se recomienda incluir alimentos como el aceite de oliva, aguacates y frutos secos, que aportan grasas saludables y contribuyen a un mejor manejo metabólico."
# obs5="Polifagia. Al estar en tratamiento con Dapagliflozina, se incrementa el riesgo de infecciones genitourinarias, lo cual se puede manejar con una dieta rica en líquidos y un consumo moderado de azúcares y carbohidratos. Esta estrategia dietética ayuda a minimizar el riesgo de infecciones y apoya la eficacia del medicamento."
# obs6="La asociación del tratamiento con Pioglitazona con el aumento de peso resalta la necesidad de una dieta equilibrada y baja en calorías. Evitar alimentos procesados y bebidas azucaradas es fundamental para controlar el aumento de peso y mantener un perfil glucémico estable."
# obs7="La estabilidad en los niveles de glucosa lograda con los inhibidores de la DPP-4 se complementa con una dieta rica en fibra y baja en carbohidratos simples. Incrementar la ingesta de vegetales, especialmente los de hoja verde, es beneficioso para el control glucémico y la salud general."
# obs8="La necesidad de ajustar la dosis de insulina en días de actividad física intensa subraya la importancia de consumir snacks ricos en carbohidratos complejos antes del ejercicio. Esto ayuda a prevenir hipoglucemias y asegura un suministro energético adecuado durante la actividad física."
# obs9="Con el tratamiento de Repaglinida, es esencial no omitir comidas para evitar hipoglucemias. Adoptar un plan de alimentación estructurado que incluya snacks saludables entre comidas asegura la estabilidad glucémica y apoya la eficacia del tratamiento."
# obs10="El tratamiento con Sulfonylureas exige una vigilancia constante de los síntomas de hipoglucemia. Disponer de snacks rápidos y saludables ricos en glucosa, como fruta fresca o jugos sin azúcar añadido, es crucial para responder de manera efectiva a cualquier signo de hipoglucemia."

# obs1_2v="Test de hemoglobina glicosilada (HbA1c) muestra una reducción significativa, indicando un control glucémico excelente. La continuación del tratamiento con Metformina, junto con una dieta baja en carbodratos refinados, ha resultado en una mejora sostenida. El paciente ha reportado una preferencia consolidada por carbohidratos complejos y una notable disminución en la tentación por alimentos altos en azúcares simples."
# obs2_2v="La adherencia al conteo de carbohidratos y la administración precisa de insulina prandial ha demostrado ser efectiva en la estabilización de los niveles de glucosa en sangre. El paciente ha desarrollado habilidades avanzadas en la planificación de comidas y la selección de alimentos, lo que ha contribuido a evitar fluctuaciones glucémicas y ha mejorado su calidad de vida significativamente."
# obs3_2v="El seguimiento riguroso de la hidratación y la limitación en el consumo de bebidas azucaradas han reforzado la eficacia de los inhibidores de SGLT2 y han mantenido la salud renal en óptimas condiciones. El paciente ha reportado un estilo de vida más activo y una mejor comprensión de la importancia del agua para su salud metabólica y general."
# obs4_2v="La dieta baja en grasas saturadas, complementada con el tratamiento con agonistas de GLP-1, ha tenido un impacto positivo adicional en la reducción de peso y la mejora del perfil metabólico. El paciente ha integrado fuentes de grasas saludables en su dieta diaria, lo que ha contribuido a un mayor bienestar y satisfacción con su tratamiento."
# obs5_2v="La incidencia de infecciones genitourinarias ha disminuido notablemente gracias a una dieta rica en líquidos y baja en azúcares. El tratamiento con Dapagliflozina sigue siendo efectivo, y el paciente ha adoptado medidas preventivas dietéticas como parte integral de su rutina diaria, lo que ha mejorado su calidad de vida."
# obs6_2v="La implementación de una dieta equilibrada y baja en calorías ha demostrado ser crucial en el manejo del aumento de peso asociado con Pioglitazona. El paciente ha logrado mantener un peso estable mediante la elección consciente de alimentos no procesados y la eliminación de bebidas azucaradas de su dieta."
# obs7_2v="El aumento en el consumo de fibra y la reducción de carbohidratos simples han fortalecido los efectos de los inhibidores de la DPP-4 en el control glucémico. El paciente ha mostrado una mejora en los indicadores de salud general, gracias a un mayor consumo de vegetales, especialmente de hoja verde, en su alimentación diaria."
# obs8_2v="La planificación de comidas antes del ejercicio se ha perfeccionado, con un enfoque en carbohidratos complejos para prevenir la hipoglucemia durante la actividad física intensa. Este enfoque ha permitido al paciente disfrutar de una mayor flexibilidad y seguridad en sus rutinas de ejercicio, optimizando los beneficios para su salud."
# obs9_2v="La estructuración de un plan de alimentación que evita la omisión de comidas ha demostrado ser efectiva en la prevención de hipoglucemias y en el soporte de la eficacia de la Repaglinida. El paciente ha adoptado hábitos alimenticios más saludables, incluyendo snacks entre comidas, lo que ha contribuido a una estabilidad glucémica constante."
# obs10_2v="La vigilancia de los síntomas de hipoglucemia se ha integrado de manera efectiva en la vida diaria del paciente, con un enfoque en tener disponibles snacks saludables ricos en glucosa. Esta práctica ha mejorado la capacidad del paciente para gestionar episodios de hipoglucemia de manera rápida y eficaz, asegurando un control glucémico óptimo con el tratamiento de Sulfonylureas."

obs1="""
⁃Eliminar el consumo de carbohidratos de absorcion rápida de la dieta(pastas, arroces, patata, cuscús, pan y otros cereales). Cuando se ingieran elegir las integrales
⁃Eliminar de la dieta zumos de fruta industriales, bebidas refrescantes conazúcar, galletería, bollería, pasteleria, dulces, azúcar de mesa, cereales dedesayuno azucarados y sustituir por avena, muesli, pan integral, etc.
⁃Aumentar el consumo de frutas y verduras.
⁃Aumentar la proporción de consumo de alimentos integrales/alimentos refinados.
⁃Aumentar el consumo de alimentos ricos en omega-3 como pescados azules y frutos secos, en especial las nueces. Tener en cuenta la cantidad de estos pues son alimentos de elevada densidad energética por lo que en función de las cantidades ingeridas podría darse una ralentización en la bajada de peso
⁃Llevar a cabo una dieta equilibrada lo que implica una gran variedad de alimentos. Debe ser un dietista-nutricionista quien prescriba dicha planificación dietética. Considerando la ausencia de contraindicaciones y la relación coste/beneficio se recomienda la bajada de peso controlada y gradual bajo supervisión de un dietista-nutricionista. Bajadas bruscas y elevadas de peso pueden a acentuar la inflamación y fibrosis hepática.
⁃Reducir el consumo de grasas saturadas como quesos curados, lácteos y derivados lácteos enteros, carnes grasas, bollería pastelería, dulces, etc.
⁃Eliminar de la dieta el consumo de alcohol.
⁃Debe realizar ejercicio físico. El ejercicio debe ser de carácter aeróbico (caminar rápido, trotar suavemente, bicicleta llaneando, patinar, bailar, elíptica,etc.). Se recomienda una frecuencia de 3 a 5 veces por semana con una duración de unos 30 minutos. En el caso de pérdida de peso se recomienda incrementar a al menos 60minutos y en condiciones de ayunoposabsortivo (una vez realizado el proceso de digestión en el estómago).
"""
obs2="DIETA BAJA EN POTASIO. SUPLEMENTOS DE NUTRICION ENTERAL SEGÚN RECOMENDACIONES VERBALES."
obs3="DIETA CONTROLADA EN HC, AZUCARES SIMPLES. SEGUIMIENTO POR NUTRICIONISTA"
obs4="CONSUMIR ENTRE 1G-1.5 G PROTEINA/KG/DIA. SEGUIMIENTO POR NUTRICIONISTA"
obs5="DIETA hipocalórica controlada en HC, AZUCARES SIMPLES. SEGUIMIENTO POR NUTRICIONISTA"
obs6="""
Implementar una dieta mediterránea: enfatizando el consumo de frutas, verduras, legumbres, frutos secos, cereales integrales y aceite de oliva como principal fuente de grasa. Limitar la ingesta de carnes rojas, prefiriendo pescados y aves.
Reducir la ingesta de sodio: evitar alimentos altamente procesados y salazones. Utilizar hierbas y especias para mejorar el sabor de los alimentos.
Incorporar alimentos ricos en calcio y vitamina D: como lácteos bajos en grasa, verduras de hoja verde y pescados grasos, para mejorar la salud ósea.
Moderación en el consumo de alcohol: limitar la ingesta a no más de una copa diaria en mujeres y dos en hombres.
Actividad física regular: combinar ejercicios aeróbicos con entrenamiento de fuerza para mejorar la salud cardiovascular y muscular. Se sugiere una duración de 30 a 60 minutos la mayoría de los días de la semana.
Seguimiento profesional: es crucial el acompañamiento por un dietista-nutricionista para personalizar la dieta según necesidades individuales y asegurar el cumplimiento de los objetivos nutricionales."""
obs7="Dieta baja en grasas saturadas y colesterol: enfocarse en alimentos vegetales, pescados, pollo sin piel y lácteos bajos en grasa. Supervisión de un nutricionista: para ajustar la dieta de acuerdo a las necesidades energéticas y nutricionales del paciente."
obs8="Dieta rica en potasio: incrementar el consumo de frutas como plátano, naranja, y verduras. Reducir alimentos procesados que suelen ser bajos en potasio. Acompañamiento nutricional: seguimiento regular con un nutricionista para ajustar la ingesta de potasio según niveles sanguíneos."
obs9="Aumentar la ingesta de proteínas de alta calidad: enfocarse en fuentes magras como pescado, pechuga de pollo, tofu y legumbres. Control y seguimiento nutricional: imprescindible para monitorizar la ingesta proteica y ajustarla según la evolución del estado de salud."
obs10="Dieta rica en antioxidantes: incrementar el consumo de alimentos ricos en antioxidantes como bayas, frutas cítricas, frutos secos, verduras de hoja verde y té verde. Seguimiento nutricional personalizado: para asegurar la adecuada incorporación de antioxidantes en la dieta y ajustar según necesidades específicas del paciente."

obs1_2v="""
Mantener la reducción en el consumo de carbohidratos de rápida absorción, observando mejoras en los niveles de glucosa sanguínea.
Introducir alternativas de cereales sin gluten (como quinua y amaranto) para diversificar la dieta y evitar posibles sensibilidades.
Fomentar el consumo de verduras de hoja verde oscura para mejorar el aporte de minerales y vitaminas.
Revisar la ingesta de omega-3, ajustando las porciones para optimizar los niveles de colesterol.
Continuar con el ejercicio aeróbico, evaluando la posibilidad de incorporar entrenamiento de fuerza para aumentar la masa muscular y mejorar el metabolismo.
Reafirmar la importancia de evitar el alcohol y las grasas saturadas, especialmente en pacientes con tendencia a enfermedades hepáticas.
Refuerzo de la educación nutricional por parte del dietista-nutricionista para asegurar la sostenibilidad del plan dietético."""
obs2_2v="Ajustar la dieta baja en potasio según los niveles actuales, potencialmente reintroduciendo alimentos restringidos previamente de forma controlada. Valorar la transición de suplementos de nutrición enteral a una dieta más variada y rica en nutrientes naturales."
obs3_2v="Adaptar la dieta controlada en hidratos de carbono y azúcares simples según la respuesta metabólica del paciente y su evolución en el control de peso. Potenciar la educación sobre el tamaño de las porciones y el conteo de carbohidratos para mejorar la autogestión de la dieta."
obs4_2v="Revisar la ingesta proteica, asegurando que se alcanza el objetivo de 1g-1.5g proteína/kg/día, ajustando según el nivel de actividad física y la masa muscular del paciente. Incluir fuentes de proteínas vegetales para promover una dieta más sostenible y variada."
obs5_2v="Modificar la dieta hipocalórica según los cambios en el metabolismo del paciente, enfocándose en mantener un balance energético adecuado para promover una pérdida de peso saludable. Intensificar el seguimiento nutricional para ajustar la ingesta de macronutrientes y prevenir deficiencias."
obs6_2v="""
Evaluar el impacto de la dieta mediterránea en los indicadores de salud cardiovascular y ajustar la ingesta de aceite de oliva y frutos secos según las necesidades calóricas.
Fomentar la diversidad en el consumo de proteínas, incorporando más fuentes vegetales.
Mantener la moderación en el consumo de alcohol y revisar la necesidad de suplementación de vitamina D, especialmente en poblaciones con limitada exposición solar."""
obs7_2v="Continuar promoviendo una dieta baja en grasas saturadas, valorando la introducción de más variedades de pescado para aumentar la ingesta de omega-3. Monitorear los niveles de colesterol y ajustar la dieta según sea necesario."
obs8_2v="Tras ajustar la dieta rica en potasio, monitorear los niveles séricos para prevenir hiperpotasemia, especialmente en pacientes con riesgo de enfermedad renal. Evaluar la inclusión de vegetales crucíferos y tubérculos cocidos para un aporte equilibrado de potasio"
obs9_2v="Aumentar la variedad de fuentes de proteínas magras, incluyendo pescado blanco y legumbres, para promover una dieta equilibrada. Considerar la evaluación de necesidades proteicas específicas en caso de cambios significativos en el peso o la composición corporal."
obs10_2v="Introducir más alimentos fermentados ricos en antioxidantes como el kéfir y el kimchi para promover la salud intestinal. Continuar con el seguimiento nutricional para evaluar la efectividad de la dieta rica en antioxidantes en la reducción del estrés oxidativo y su impacto en la salud general."


observaciones_1visita = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10]
observaciones_2visita = [obs1_2v, obs2_2v, obs3_2v, obs4_2v, obs5_2v, obs6_2v, obs7_2v, obs8_2v, obs9_2v, obs10_2v]

# Si el DataFrame no tiene exactamente 10 filas, este paso fallará
if len(df_ejemplo) == 10:
    df_ejemplo['observaciones_v1'] = observaciones_1visita
    df_ejemplo['observaciones_v2'] = observaciones_2visita
else:
    print("El DataFrame no tiene 10 filas. Tiene {} filas.".format(len(df_ejemplo)))

df_ejemplo.to_csv('./archivos/datos_con_10_ejemplos_v5.csv', index=False)