# %%
import os
import random
import sys

import numpy as np
import torch

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
# Seed for reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)

# %%

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sonar_encoder_decoder import SonarEncoderDecoder

# %%
english_encoder_decoder = SonarEncoderDecoder(
    device="cuda", encoder_language="eng_Latn", decoder_language="eng_Latn"
)
spanish_encoder_decoder = SonarEncoderDecoder(
    device="cuda", encoder_language="spa_Latn", decoder_language="spa_Latn"
)


# %%
english_sequence = ("dog",)
engligh_dog_embedding = english_encoder_decoder.encode(
    english_encoder_decoder.list_str_to_token_ids(english_sequence).unsqueeze(0)
)[0]
# %%
spanish_sequence = ("perro",)
spanish_dog_embedding = spanish_encoder_decoder.encode(
    spanish_encoder_decoder.list_str_to_token_ids(spanish_sequence).unsqueeze(0)
)[0]

# %%
engligh_dog_embedding - spanish_dog_embedding
# %%
# Calculate cosine similarity between English and Spanish embeddings
cosine_similarity = torch.nn.functional.cosine_similarity(
    engligh_dog_embedding, spanish_dog_embedding
)
print(
    f"Cosine similarity between 'dog' and 'perro' embeddings: {cosine_similarity.item():.4f}"
)

# %%
english_encoder_decoder.token_ids_to_list_str_batch(
    english_encoder_decoder.decode(engligh_dog_embedding.unsqueeze(0))
)
# %%
english_encoder_decoder.token_ids_to_list_str_batch(
    english_encoder_decoder.decode(spanish_dog_embedding.unsqueeze(0))
)
# %%
spanish_encoder_decoder.token_ids_to_list_str_batch(
    spanish_encoder_decoder.decode(engligh_dog_embedding.unsqueeze(0))
)
# %%
spanish_encoder_decoder.token_ids_to_list_str_batch(
    spanish_encoder_decoder.decode(spanish_dog_embedding.unsqueeze(0))
)
# %%
english_to_spanish_vector = spanish_dog_embedding - engligh_dog_embedding
english_to_spanish_vector
# %%
engligh_cat_embedding = english_encoder_decoder.encode(
    english_encoder_decoder.list_str_to_token_ids(("cat",)).unsqueeze(0)
)[0]
spanish_cat_embedding = spanish_encoder_decoder.encode(
    spanish_encoder_decoder.list_str_to_token_ids(("gato",)).unsqueeze(0)
)[0]

cosine_similarity = torch.nn.functional.cosine_similarity(
    spanish_cat_embedding, engligh_cat_embedding + english_to_spanish_vector
)
print(
    f"Cosine similarity between 'gato' and 'cat' transformed to spanish embeddings: {cosine_similarity.item():.4f}"
)
# %%

english_sentence = ("the", "dog", "and", "the", "cat")
spanish_sentence = ("el", "perro", "y", "el", "gato")

english_sentence_embedding = english_encoder_decoder.encode(
    english_encoder_decoder.list_str_to_token_ids(english_sentence).unsqueeze(0)
)[0]
spanish_sentence_embedding = spanish_encoder_decoder.encode(
    spanish_encoder_decoder.list_str_to_token_ids(spanish_sentence).unsqueeze(0)
)[0]

english_to_spanish_vector = spanish_sentence_embedding - english_sentence_embedding
english_to_spanish_vector
cosine_similarity = torch.nn.functional.cosine_similarity(
    spanish_encoder_decoder.encode(
        spanish_encoder_decoder.list_str_to_token_ids(
            ("la", "casa", "y", "el", "coche")
        ).unsqueeze(0)
    )[0],
    english_encoder_decoder.encode(
        english_encoder_decoder.list_str_to_token_ids(
            ("the", "house", "and", "the", "car")
        ).unsqueeze(0)
    )[0]
    + english_to_spanish_vector,
)
cosine_similarity

# %%
english_spanish_pairs = [
    ("the dog runs fast", "el perro corre rápido"),
    ("a cat sleeps quietly", "un gato duerme tranquilamente"),
    ("my house is big", "mi casa es grande"),
    ("she walks to school", "ella camina a la escuela"),
    ("they eat lunch together", "ellos almuerzan juntos"),
    ("he reads many books", "él lee muchos libros"),
    ("we play in the park", "jugamos en el parque"),
    ("birds fly in the sky", "los pájaros vuelan en el cielo"),
    ("fish swim in the sea", "los peces nadan en el mar"),
    ("children laugh and sing", "los niños ríen y cantan"),
    ("cars drive on roads", "los coches conducen por las carreteras"),
    ("trees grow very tall", "los árboles crecen muy altos"),
    ("flowers bloom in spring", "las flores florecen en primavera"),
    ("students study hard", "los estudiantes estudian mucho"),
    ("the moon shines bright", "la luna brilla intensamente"),
    ("rain falls from clouds", "la lluvia cae de las nubes"),
    ("wind blows through leaves", "el viento sopla entre las hojas"),
    ("people walk downtown", "la gente camina por el centro"),
    ("music fills the room", "la música llena la habitación"),
    ("snow covers the ground", "la nieve cubre el suelo"),
    ("waves crash on shore", "las olas rompen en la orilla"),
    ("sun sets in the west", "el sol se pone en el oeste"),
    ("farmers grow their crops", "los agricultores cultivan sus cosechas"),
    ("trains move on tracks", "los trenes se mueven por las vías"),
    ("planes soar overhead", "los aviones vuelan por encima"),
    ("food cooks in kitchen", "la comida se cocina en la cocina"),
    ("paint dries on walls", "la pintura se seca en las paredes"),
    ("grass grows in fields", "la hierba crece en los campos"),
    ("stars twinkle above", "las estrellas brillan arriba"),
    ("rivers flow downstream", "los ríos fluyen río abajo"),
    ("mountains rise high", "las montañas se elevan alto"),
    ("books line shelves", "los libros alinean los estantes"),
    ("clocks tick steadily", "los relojes marcan constantemente"),
    ("doors open wide", "las puertas se abren ampliamente"),
    ("windows let light in", "las ventanas dejan entrar la luz"),
    ("phones ring loudly", "los teléfonos suenan fuertemente"),
    ("computers run programs", "las computadoras ejecutan programas"),
    ("teachers help students", "los maestros ayudan a estudiantes"),
    ("doctors treat patients", "los médicos tratan pacientes"),
    ("artists paint pictures", "los artistas pintan cuadros"),
    ("writers create stories", "los escritores crean historias"),
    ("dancers move gracefully", "los bailarines se mueven con gracia"),
    ("singers perform songs", "los cantantes interpretan canciones"),
    ("athletes train hard", "los atletas entrenan duro"),
    ("chefs cook meals", "los chefs cocinan comidas"),
    ("gardeners plant flowers", "los jardineros plantan flores"),
    ("builders construct homes", "los constructores construyen casas"),
    ("drivers follow roads", "los conductores siguen carreteras"),
    ("pilots fly planes", "los pilotos vuelan aviones"),
    ("sailors navigate ships", "los marineros navegan barcos"),
    ("soldiers march forward", "los soldados marchan adelante"),
    ("firefighters save lives", "los bomberos salvan vidas"),
    ("police protect people", "la policía protege personas"),
    ("judges make decisions", "los jueces toman decisiones"),
    ("lawyers argue cases", "los abogados argumentan casos"),
    ("scientists conduct research", "los científicos realizan investigación"),
    ("engineers solve problems", "los ingenieros resuelven problemas"),
    ("architects design buildings", "los arquitectos diseñan edificios"),
    ("photographers take pictures", "los fotógrafos toman fotos"),
    ("mechanics fix cars", "los mecánicos arreglan coches"),
    ("bakers make bread", "los panaderos hacen pan"),
    ("butchers cut meat", "los carniceros cortan carne"),
    ("farmers harvest crops", "los granjeros cosechan cultivos"),
    ("fishermen catch fish", "los pescadores atrapan peces"),
    ("miners dig deep", "los mineros cavan profundo"),
    ("loggers cut trees", "los leñadores cortan árboles"),
    ("weavers make cloth", "los tejedores hacen tela"),
    ("potters shape clay", "los alfareros moldean arcilla"),
    ("smiths forge metal", "los herreros forjan metal"),
    ("carpenters build furniture", "los carpinteros construyen muebles"),
    ("plumbers fix pipes", "los fontaneros arreglan tuberías"),
    ("electricians wire homes", "los electricistas cablean casas"),
    ("painters color walls", "los pintores colorean paredes"),
    ("sculptors carve stone", "los escultores tallan piedra"),
    ("jewelers make rings", "los joyeros hacen anillos"),
    ("tailors sew clothes", "los sastres cosen ropa"),
    ("barbers cut hair", "los barberos cortan pelo"),
    ("dentists clean teeth", "los dentistas limpian dientes"),
    ("nurses help patients", "las enfermeras ayudan pacientes"),
    ("pharmacists fill prescriptions", "los farmacéuticos llenan recetas"),
    ("librarians organize books", "los bibliotecarios organizan libros"),
    ("cashiers count money", "los cajeros cuentan dinero"),
    ("servers bring food", "los meseros traen comida"),
    ("cleaners tidy rooms", "los limpiadores ordenan habitaciones"),
    ("guards watch doors", "los guardias vigilan puertas"),
    ("guides lead tours", "los guías dirigen tours"),
    ("actors perform plays", "los actores interpretan obras"),
    ("musicians play instruments", "los músicos tocan instrumentos"),
    ("comedians tell jokes", "los comediantes cuentan chistes"),
    ("magicians perform tricks", "los magos realizan trucos"),
    ("athletes compete games", "los atletas compiten juegos"),
    ("coaches train teams", "los entrenadores entrenan equipos"),
    ("referees call fouls", "los árbitros marcan faltas"),
    ("fans cheer loudly", "los aficionados animan fuertemente"),
    ("reporters write news", "los reporteros escriben noticias"),
    ("editors check stories", "los editores revisan historias"),
    ("producers make films", "los productores hacen películas"),
    ("directors guide actors", "los directores guían actores"),
    ("designers create clothes", "los diseñadores crean ropa"),
    ("models wear fashion", "los modelos visten moda"),
    ("critics review art", "los críticos revisan arte"),
    ("curators manage museums", "los curadores manejan museos"),
    ("professors teach classes", "los profesores enseñan clases"),
    ("students learn lessons", "los estudiantes aprenden lecciones"),
    ("researchers study topics", "los investigadores estudian temas"),
    ("analysts examine data", "los analistas examinan datos"),
    ("programmers write code", "los programadores escriben código"),
    ("technicians fix computers", "los técnicos arreglan computadoras"),
    ("operators answer phones", "los operadores contestan teléfonos"),
    ("secretaries organize offices", "las secretarias organizan oficinas"),
    ("managers lead teams", "los gerentes lideran equipos"),
    ("executives make decisions", "los ejecutivos toman decisiones"),
    ("salespeople sell products", "los vendedores venden productos"),
    ("customers buy goods", "los clientes compran bienes"),
    ("bankers handle money", "los banqueros manejan dinero"),
    ("accountants track finances", "los contadores rastrean finanzas"),
    ("investors buy stocks", "los inversores compran acciones"),
    ("traders sell shares", "los comerciantes venden acciones"),
    ("brokers make deals", "los corredores hacen tratos"),
    ("agents sell houses", "los agentes venden casas"),
    ("renters pay rent", "los inquilinos pagan renta"),
    ("landlords own buildings", "los propietarios poseen edificios"),
    ("neighbors live nearby", "los vecinos viven cerca"),
    ("friends meet often", "los amigos se encuentran frecuentemente"),
    ("families gather together", "las familias se reúnen juntas"),
    ("parents raise children", "los padres crían niños"),
    ("babies cry loudly", "los bebés lloran fuertemente"),
    ("children play games", "los niños juegan juegos"),
    ("teenagers study subjects", "los adolescentes estudian materias"),
    ("adults work jobs", "los adultos trabajan empleos"),
    ("seniors enjoy retirement", "los mayores disfrutan jubilación"),
    ("couples walk together", "las parejas caminan juntas"),
    ("dogs bark loudly", "los perros ladran fuertemente"),
    ("cats chase mice", "los gatos persiguen ratones"),
    ("birds build nests", "los pájaros construyen nidos"),
    ("squirrels gather nuts", "las ardillas recolectan nueces"),
    ("rabbits hop quickly", "los conejos saltan rápidamente"),
    ("deer run fast", "los ciervos corren rápido"),
    ("bears catch fish", "los osos atrapan peces"),
    ("wolves hunt prey", "los lobos cazan presas"),
    ("lions roar loudly", "los leones rugen fuertemente"),
    ("tigers stalk quietly", "los tigres acechan silenciosamente"),
    ("elephants walk slowly", "los elefantes caminan lentamente"),
    ("monkeys climb trees", "los monos trepan árboles"),
    ("snakes slither silently", "las serpientes se deslizan silenciosamente"),
    ("frogs jump high", "las ranas saltan alto"),
    ("fish swim deep", "los peces nadan profundo"),
    ("sharks hunt fish", "los tiburones cazan peces"),
    ("whales breach water", "las ballenas rompen agua"),
    ("dolphins play waves", "los delfines juegan olas"),
    ("seals rest beaches", "las focas descansan playas"),
    ("crabs walk sideways", "los cangrejos caminan lateralmente"),
    ("insects buzz around", "los insectos zumban alrededor"),
    ("bees make honey", "las abejas hacen miel"),
    ("ants work together", "las hormigas trabajan juntas"),
    ("spiders spin webs", "las arañas tejen redes"),
    ("butterflies float gently", "las mariposas flotan suavemente"),
    ("flowers grow tall", "las flores crecen altas"),
    ("trees provide shade", "los árboles proporcionan sombra"),
    ("grass covers ground", "la hierba cubre suelo"),
    ("vegetables grow gardens", "las verduras crecen jardines"),
    ("fruits ripen trees", "las frutas maduran árboles"),
    ("leaves change colors", "las hojas cambian colores"),
    ("seeds sprout ground", "las semillas brotan suelo"),
    ("roots grow deep", "las raíces crecen profundo"),
    ("branches sway wind", "las ramas se mecen viento"),
    ("vines climb walls", "las enredaderas trepan paredes"),
    ("moss grows rocks", "el musgo crece rocas"),
    ("mushrooms grow forest", "los hongos crecen bosque"),
    ("weeds spread quickly", "las malezas se propagan rápidamente"),
    ("crops grow fields", "los cultivos crecen campos"),
    ("wheat waves wind", "el trigo ondea viento"),
    ("corn grows tall", "el maíz crece alto"),
    ("rice grows water", "el arroz crece agua"),
    ("potatoes grow underground", "las patatas crecen subterráneo"),
    ("tomatoes ripen vines", "los tomates maduran enredaderas"),
    ("peppers grow plants", "los pimientos crecen plantas"),
    ("carrots grow soil", "las zanahorias crecen suelo"),
    ("onions grow ground", "las cebollas crecen suelo"),
    ("lettuce grows gardens", "la lechuga crece jardines"),
    ("cabbage forms heads", "el repollo forma cabezas"),
    ("beans climb poles", "los frijoles trepan postes"),
    ("peas grow pods", "los guisantes crecen vainas"),
    ("apples grow trees", "las manzanas crecen árboles"),
    ("oranges grow trees", "las naranjas crecen árboles"),
    ("bananas grow clusters", "los plátanos crecen racimos"),
    ("grapes grow vines", "las uvas crecen enredaderas"),
    ("strawberries grow plants", "las fresas crecen plantas"),
    ("blueberries grow bushes", "los arándanos crecen arbustos"),
    ("raspberries grow canes", "las frambuesas crecen cañas"),
    ("cherries grow trees", "las cerezas crecen árboles"),
    ("peaches grow trees", "los melocotones crecen árboles"),
    ("pears grow trees", "las peras crecen árboles"),
    ("plums grow trees", "las ciruelas crecen árboles"),
    ("lemons grow trees", "los limones crecen árboles"),
    ("limes grow trees", "las limas crecen árboles"),
    ("figs grow trees", "los higos crecen árboles"),
    ("dates grow palms", "los dátiles crecen palmeras"),
    ("coconuts grow palms", "los cocos crecen palmeras"),
    ("nuts grow trees", "las nueces crecen árboles"),
    ("almonds grow trees", "las almendras crecen árboles"),
    ("walnuts grow trees", "las nueces crecen árboles"),
    ("pecans grow trees", "los pacanas crecen árboles"),
    ("chestnuts grow trees", "las castañas crecen árboles"),
    ("peanuts grow ground", "los cacahuetes crecen suelo"),
    ("sunflowers grow tall", "los girasoles crecen altos"),
    ("roses bloom gardens", "las rosas florecen jardines"),
    ("tulips bloom spring", "los tulipanes florecen primavera"),
    ("daisies grow fields", "las margaritas crecen campos"),
    ("lilies grow gardens", "los lirios crecen jardines"),
    ("orchids grow trees", "las orquídeas crecen árboles"),
    ("violets grow shade", "las violetas crecen sombra"),
    ("pansies grow gardens", "los pensamientos crecen jardines"),
    ("marigolds grow gardens", "las caléndulas crecen jardines"),
    ("daffodils grow spring", "los narcisos crecen primavera"),
    ("iris grow gardens", "los iris crecen jardines"),
    ("chrysanthemums grow fall", "los crisantemos crecen otoño"),
    ("poinsettias grow winter", "las nochebuenas crecen invierno"),
    ("cacti grow desert", "los cactus crecen desierto"),
    ("bamboo grows fast", "el bambú crece rápido"),
    ("ferns grow shade", "los helechos crecen sombra"),
    ("palms grow beaches", "las palmeras crecen playas"),
    ("pines grow mountains", "los pinos crecen montañas"),
    ("oaks grow strong", "los robles crecen fuertes"),
    ("maples grow tall", "los arces crecen altos"),
    ("willows grow water", "los sauces crecen agua"),
    ("birch grows forest", "el abedul crece bosque"),
    ("cedars grow high", "los cedros crecen alto"),
    ("redwoods grow huge", "las secoyas crecen enormes"),
    ("eucalyptus grows tall", "el eucalipto crece alto"),
    ("aspen grows mountains", "el álamo crece montañas"),
    ("cypress grows swamps", "el ciprés crece pantanos"),
    ("magnolia grows south", "la magnolia crece sur"),
    ("dogwood grows spring", "el cornejo crece primavera"),
    ("holly grows winter", "el acebo crece invierno"),
    ("juniper grows low", "el enebro crece bajo"),
    ("spruce grows north", "el abeto crece norte"),
    ("hemlock grows forest", "la cicuta crece bosque"),
    ("larch grows mountains", "el alerce crece montañas"),
    ("beech grows woods", "el haya crece bosques"),
    ("elm grows parks", "el olmo crece parques"),
    ("ash grows tall", "el fresno crece alto"),
    ("sycamore grows large", "el sicomoro crece grande"),
    ("cottonwood grows rivers", "el álamo crece ríos"),
    ("locust grows fast", "la acacia crece rápido"),
    ("catalpa grows flowers", "la catalpa crece flores"),
    ("mulberry grows fruit", "la morera crece fruta"),
    ("persimmon grows sweet", "el caqui crece dulce"),
    ("pawpaw grows shade", "la papaya crece sombra"),
    ("olive grows mediterranean", "el olivo crece mediterráneo"),
    ("pomegranate grows warm", "la granada crece cálido"),
    ("quince grows fall", "el membrillo crece otoño"),
    ("kiwi grows vines", "el kiwi crece enredaderas"),
    ("passion grows vines", "la pasión crece enredaderas"),
    ("guava grows tropical", "la guayaba crece tropical"),
    ("mango grows trees", "el mango crece árboles"),
    ("papaya grows palms", "la papaya crece palmeras"),
    ("avocado grows trees", "el aguacate crece árboles"),
    ("lychee grows trees", "el lichi crece árboles"),
    ("durian grows large", "el durián crece grande"),
    ("jackfruit grows huge", "la jaca crece enorme"),
    ("breadfruit grows trees", "el árbol del pan crece árboles"),
    ("soursop grows tropical", "la guanábana crece tropical"),
    ("carambola grows star", "la carambola crece estrella"),
    ("rambutan grows hairy", "el rambután crece peludo"),
    ("dragonfruit grows cactus", "la pitaya crece cactus"),
    ("tamarind grows pods", "el tamarindo crece vainas"),
    ("noni grows islands", "el noni crece islas"),
    ("sapodilla grows brown", "el chicozapote crece marrón"),
    ("custard grows sweet", "la chirimoya crece dulce"),
    ("miracle grows berries", "el milagro crece bayas"),
    ("acai grows palms", "el açaí crece palmeras"),
    ("goji grows berries", "el goji crece bayas"),
    ("nectarine grows smooth", "la nectarina crece suave"),
    ("kumquat grows small", "el kumquat crece pequeño"),
    ("plantain grows bananas", "el plátano crece bananos"),
    ("starfruit grows tropical", "la carambola crece tropical"),
    ("ugli grows citrus", "el ugli crece cítricos"),
    ("yuzu grows citrus", "el yuzu crece cítricos"),
    ("calamansi grows small", "el calamansi crece pequeño"),
    ("feijoa grows green", "la feijoa crece verde"),
    ("jabuticaba grows trunk", "la jabuticaba crece tronco"),
    ("longan grows sweet", "el longan crece dulce"),
    ("mamey grows tropical", "el mamey crece tropical"),
    ("mangosteen grows purple", "el mangostán crece púrpura"),
    ("salak grows clusters", "el salak crece racimos"),
    ("santol grows yellow", "el santol crece amarillo"),
    ("sapote grows orange", "el zapote crece naranja"),
    ("soursop grows green", "la guanábana crece verde"),
    ("sugar grows apples", "el azúcar crece manzanas"),
    ("tangelo grows orange", "el tangelo crece naranja"),
    ("ugli grows fruit", "el ugli crece fruta"),
    ("wampee grows clusters", "el wampee crece racimos"),
    ("white grows sapote", "el blanco crece zapote"),
    ("wood grows apple", "la madera crece manzana"),
    ("yangmei grows red", "el yangmei crece rojo"),
    ("yellow grows passion", "el amarillo crece pasión"),
    ("ziziphus grows jujube", "el azufaifo crece jujube"),
]
english_to_spanish_vectors = []
english_vectors = []
spanish_vectors = []
for english_text, spanish_text in english_spanish_pairs:
    english_tokens = english_encoder_decoder.tokenizer_encoder(english_text)
    spanish_tokens = spanish_encoder_decoder.tokenizer_encoder(spanish_text)

    english_token_ids = english_tokens[1:-1]  # Remove BOS/EOS tokens
    spanish_token_ids = spanish_tokens[1:-1]  # Remove BOS/EOS tokens

    english_embedding = english_encoder_decoder.encode(english_token_ids.unsqueeze(0))[
        0
    ]
    spanish_embedding = spanish_encoder_decoder.encode(spanish_token_ids.unsqueeze(0))[
        0
    ]

    english_to_spanish_vectors.append(spanish_embedding - english_embedding)
    english_vectors.append(english_embedding)
    spanish_vectors.append(spanish_embedding)

# %%
# english_to_spanish_vector = (
#     torch.stack(english_to_spanish_vectors, dim=0).squeeze().mean(dim=0)
# )
# Convert lists of vectors to tensors and stack them
english_vectors_tensor = torch.stack(english_vectors, dim=0).squeeze()
spanish_vectors_tensor = torch.stack(spanish_vectors, dim=0).squeeze()

spanish_vectors_tensor - english_vectors_tensor
english_to_spanish_vector = (spanish_vectors_tensor - english_vectors_tensor).mean(
    dim=0
)
english_to_spanish_vector
# %%
english_sentence = ("the", "dog", "and", "the", "cat")
spanish_sentence = ("el", "perro", "y", "el", "gato")

english_sentence_embedding = english_encoder_decoder.encode(
    english_encoder_decoder.list_str_to_token_ids(english_sentence).unsqueeze(0)
)[0]
spanish_sentence_embedding = spanish_encoder_decoder.encode(
    spanish_encoder_decoder.list_str_to_token_ids(spanish_sentence).unsqueeze(0)
)[0]

cosine_similarity = torch.nn.functional.cosine_similarity(
    spanish_encoder_decoder.encode(
        spanish_encoder_decoder.list_str_to_token_ids(
            ("la", "casa", "y", "el", "coche")
        ).unsqueeze(0)
    )[0],
    english_encoder_decoder.encode(
        english_encoder_decoder.list_str_to_token_ids(
            ("the", "house", "and", "the", "car")
        ).unsqueeze(0)
    )[0],
)
print(cosine_similarity)

# english_to_spanish_vector = spanish_sentence_embedding - english_sentence_embedding
# english_to_spanish_vector
cosine_similarity = torch.nn.functional.cosine_similarity(
    spanish_encoder_decoder.encode(
        spanish_encoder_decoder.list_str_to_token_ids(
            ("la", "casa", "y", "el", "coche")
        ).unsqueeze(0)
    )[0],
    english_encoder_decoder.encode(
        english_encoder_decoder.list_str_to_token_ids(
            ("the", "house", "and", "the", "car")
        ).unsqueeze(0)
    )[0]
    + english_to_spanish_vector,
)
print(cosine_similarity)
# %%

# Combine English and Spanish vectors
all_vectors = torch.cat(
    [
        english_vectors_tensor[: english_vectors_tensor.shape[0] // 1],
        spanish_vectors_tensor[: spanish_vectors_tensor.shape[0] // 1],
    ]
)

# Convert to numpy for PCA
all_vectors_np = all_vectors.cpu().numpy()


# translation_vector_np = english_to_spanish_vector.unsqueeze(0).cpu().numpy()
mean_english_vector_np = english_vectors_tensor.mean(dim=0).unsqueeze(0).cpu().numpy()
mean_spanish_vector_np = spanish_vectors_tensor.mean(dim=0).unsqueeze(0).cpu().numpy()

# Perform PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(all_vectors_np)
translation_vector_2d_destination = pca.transform(mean_spanish_vector_np)
translation_vector_2d_origin = pca.transform(mean_english_vector_np)

# Split back into English and Spanish vectors
n_pairs = len(english_vectors)
# english_2d = vectors_2d[:n_pairs]
# spanish_2d = vectors_2d[n_pairs:]
english_2d = pca.transform(english_vectors_tensor.cpu().numpy())
spanish_2d = pca.transform(spanish_vectors_tensor.cpu().numpy())

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(english_2d[:, 0], english_2d[:, 1], c="blue", label="English", alpha=0.6)
plt.scatter(spanish_2d[:, 0], spanish_2d[:, 1], c="red", label="Spanish", alpha=0.6)

# Draw lines connecting corresponding pairs
for i in range(n_pairs):
    plt.plot(
        [english_2d[i, 0], spanish_2d[i, 0]],
        [english_2d[i, 1], spanish_2d[i, 1]],
        "gray",
        alpha=0.3,
    )

# Plot the translation vector
plt.quiver(
    translation_vector_2d_origin[0, 0],
    translation_vector_2d_origin[0, 1],
    translation_vector_2d_destination[0, 0] - translation_vector_2d_origin[0, 0],
    translation_vector_2d_destination[0, 1] - translation_vector_2d_origin[0, 1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="green",
    label="Translation Vector",
    width=0.01,
)

plt.title("PCA of English and Spanish Word Embeddings")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%

# Test cosine similarity for various sentence pairs
test_sentence_pairs = [
    (
        ("el", "perro", "corre", "en", "el", "parque"),
        ("the", "dog", "runs", "in", "the", "park"),
    ),
    (("yo", "voy", "al", "mar"), ("i", "go", "to", "sea")),
    (("el", "sol", "da", "luz"), ("the", "sun", "gives", "light")),
    (("mi", "pan", "es", "sal"), ("my", "bread", "is", "salt")),
    (("tu", "ves", "la", "flor"), ("you", "see", "the", "rose")),
]

print("\nTesting translation vector on new sentences:")
print("-" * 50)

for spanish_sentence, english_sentence in test_sentence_pairs:
    # Encode sentences once
    spanish_embedding = spanish_encoder_decoder.encode(
        spanish_encoder_decoder.list_str_to_token_ids(spanish_sentence).unsqueeze(0)
    )[0]
    english_embedding = english_encoder_decoder.encode(
        english_encoder_decoder.list_str_to_token_ids(english_sentence).unsqueeze(0)
    )[0]
    translated_english = english_embedding + english_to_spanish_vector

    # Calculate cosine similarities
    cosine_similarity = torch.nn.functional.cosine_similarity(
        spanish_embedding, english_embedding
    )
    translated_cosine_similarity = torch.nn.functional.cosine_similarity(
        spanish_embedding, translated_english
    )

    # Calculate L2 distances
    distance = torch.norm(spanish_embedding - english_embedding)
    translated_distance = torch.norm(spanish_embedding - translated_english)

    print(f"\nSpanish: {' '.join(spanish_sentence)}")
    print(f"English: {' '.join(english_sentence)}")
    print(f"Original cosine similarity: {cosine_similarity.item():.4f}")
    print(f"Similarity after translation: {translated_cosine_similarity.item():.4f}")
    print(
        f"Improvement: {(translated_cosine_similarity - cosine_similarity).item():.4f}"
    )
    print(f"Original L2 distance: {distance.item():.4f}")
    print(f"Distance after translation: {translated_distance.item():.4f}")
    print(f"Distance improvement: {(distance - translated_distance).item():.4f}")

# %%
