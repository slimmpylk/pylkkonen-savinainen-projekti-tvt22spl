import math
import json

# Load JSON data
with open('updated_data.json', 'r') as file:
    data = json.load(file)

x_train_normal = [[item["sensorvalue_b"], item["sensorvalue_c"], item["sensorvalue_d"]] for item in data]

# Dummy labels (replace this with actual labels)
y_train_normal = [item["sensorvalue_a"] for item in data]

def relu(x):
    # ReLU palauttaa syötteen arvon, jos se on positiivinen, ja nollan, jos se on negatiivinen.
    # KÄYTETÄÄN "gradientin katoamiseksi" syvissä verkoissa, ja se on laskennallisesti yksinkertaisempi kuin esimerkiksi sigmoid- tai tanh-funktiot.
    return max(0.0, x)

def softmax(x):
    # Softmax muuntaa vektorin (joka voi sisältää mitä tahansa reaalisia lukuja) todennäköisyysjakaumaksi. 
    # varmistaa, että kaikkien elementtien summa on 1, ja jokainen elementti on arvoltaan 0:n ja 1:n välillä.
    e_x = [math.exp(i - max(x)) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

# Given weights and biases
weights_layer_0 = [
    0.3036405146121979, 0.24570688605308533, 0.145527184009552, -0.45107346773147583, 
    -0.20215854048728943, -0.04948733374476433, -0.029027696698904037, -0.45735663175582886, 
    0.3682192265987396, 0.07978120446205139, -0.20779955387115479, -0.38776350021362305, 
    -0.39982593059539795, -0.1261535882949829, -0.19824664294719696, 0.06371012330055237, 
    -0.35693883895874023, -0.34639430046081543, 0.2010888308286667, -0.42040136456489563, 
    0.104937344789505, -0.13197436928749084, -0.08856919407844543, -0.38764411211013794, 
    0.059484731405973434, 0.1955423653125763, -0.26626044511795044, -0.023263877257704735, 
    0.27822211384773254, 0.17185768485069275, -0.35959452390670776, -0.3598894476890564, 
    -0.2824243903160095, -0.1250041425228119, 0.3915944993495941, 0.06559625267982483, 
    0.03830006718635559, -0.28623056411743164, -0.4329483211040497, -0.14602723717689514, 
    -0.0009583384962752461, 0.38147351145744324, 0.3470211625099182, -0.4701763689517975, 
    -0.4709697961807251, -0.32061636447906494, -0.21941912174224854, 0.07032915949821472, 
    -0.20024406909942627, 0.07351060956716537, -0.29237422347068787, 0.22572527825832367, 
    -0.24106347560882568, -0.08797353506088257, 0.35247424244880676, -0.007615178823471069, 
    -0.08838523179292679, -0.19637644290924072, 0.11068207025527954, 0.1363341510295868, 
    -0.1658509075641632, 0.015379160642623901, -0.2037406712770462, -0.14090460538864136, 
    -0.5206899046897888, -0.36063021421432495
]

biases_layer_0 = [
    0.0, 0.0, -0.10526704788208008, 0.0, 0.0, -0.25725141167640686, 
    -0.1598031222820282, 0.016801742836833, 0.0, 0.0, 0.0, 0.0, 
    -0.03994191437959671, 0.0, -0.07373754680156708, 0.0, 0.0, 0.0, 
    -0.12127702683210373, 0.0, -0.06894215196371078, 0.0
]
weights_layer_2 = [-0.4424046576023102, 0.44084250926971436, 0.07702809572219849, -0.43659210205078125, -0.053778767585754395, -0.04353964328765869, -0.44143882393836975, 0.11249411106109619, 0.029383838176727295, -0.26440566778182983, 0.08480340242385864, 0.02446877956390381, 0.24324822425842285, 0.3116787075996399, -0.023866135627031326, -0.24606290459632874, 0.31228962540626526, -0.3075824975967407, -0.16022634506225586, -0.14252004027366638, 0.11694884300231934, 0.20088410377502441, -0.27686989307403564, -0.35739749670028687, -0.38767462968826294, 0.2953444719314575, 0.44506126642227173, 0.19526809453964233, 0.3271276354789734, -0.16063722968101501, -0.05016192048788071, -0.18065793812274933, 0.08678044378757477, -0.15127032995224, -0.18875980377197266, 0.10685499012470245, 0.0012856441317126155, -0.18572410941123962, -0.3292396068572998, 0.38250258564949036, -0.11287606507539749, -0.19856831431388855, 0.4633263647556305, -0.1761886328458786, -0.25093457102775574, -0.4350983202457428, -0.14724372327327728, -0.15742874145507812, -0.1954023241996765, 0.2998465299606323, 0.3649037480354309, -0.3735803961753845, -0.10821762681007385, 0.18028199672698975, -0.12077754735946655, 0.1077035665512085, 0.21127408742904663, -0.08662071824073792, -0.28928279876708984, 0.3451104164123535, -0.15302449464797974, -0.13402891159057617, 0.05836987495422363, -0.14853248000144958, 0.22795706987380981, -0.3153206706047058, 0.3052860498428345, -0.3888515830039978, 0.11564505100250244, -0.36346864700317383, 0.06829208135604858, 0.006894916296005249, 0.2612857222557068, 0.034804556518793106, 0.32042160630226135, 0.11562849581241608, 0.36020293831825256, 0.3468048572540283, 0.2969437837600708, 0.27903950214385986, 0.34419578313827515, 0.10937082767486572, 0.22033685445785522, 0.17138713598251343, -0.15806743502616882, 0.22218191623687744, 0.15800312161445618, 0.0765981674194336, 0.11728585511445999, 0.4133191406726837, -0.09842029213905334, 0.3442125916481018, -0.052541881799697876, 0.17334413528442383, -0.08334767818450928, -0.28429436683654785, 0.11638796329498291, 0.09368008375167847, 0.1503257155418396, 0.23873579502105713, 0.28531956672668457, 0.07805430889129639, -0.10158571600914001, -0.4326690137386322, 0.416534423828125, 0.027358591556549072, 0.02128702402114868, 0.12928426265716553, -0.27185097336769104, 0.1537780910730362, -0.3909333348274231, -0.16267596185207367, 0.12906098365783691, -0.34799209237098694, 0.2958993911743164, 0.21113282442092896, 0.3876293897628784, -0.25001323223114014, -0.11623552441596985, 0.30440694093704224, 0.0928691178560257, -0.05533836781978607, 0.39085233211517334, 0.10745985060930252, 0.28592175245285034, 0.1778510957956314, -0.30951637029647827, -0.08690589666366577, 0.1902216076850891, 0.0532151460647583, 0.1350029706954956, 0.23407769203186035]
biases_layer_2 = [-0.022157272323966026, 0.007897702977061272, 0.07932887226343155, 0.10266172140836716, -0.14710038900375366, -0.02624308317899704]

# ^^ voi olla vähän eri koska neuraltrain antaa aina uudet arvot.

def multiply_and_add(weight, input, current_sum):
    return current_sum + weight * input

def calculate_activation(weights, bias, inputs):
    activation_sum = 0
    for weight, input in zip(weights, inputs):
        activation_sum = multiply_and_add(weight, input, activation_sum)
    activation_sum += bias  # Add the bias at the end
    return activation_sum

def forward_pass(input_data):
    # First Layer
    layer_1_output = []
    for i in range(len(biases_layer_0)):
        neuron_weights = []
        # Manually select the weights for each neuron in the first layer
        for j in range(i, len(weights_layer_0), len(biases_layer_0)):
            neuron_weights.append(weights_layer_0[j])

        neuron_bias = biases_layer_0[i]
        # Calculate the activation for this neuron
        activation = calculate_activation(neuron_weights, neuron_bias, input_data)
        # Apply ReLU activation function
        layer_1_output.append(relu(activation))

    # Second Layer
    final_output = []
    for i in range(len(biases_layer_2)):
        neuron_weights = []
        # Manually select the weights for each neuron in the second layer
        for j in range(i, len(weights_layer_2), len(biases_layer_2)):
            neuron_weights.append(weights_layer_2[j])

        neuron_bias = biases_layer_2[i]
        # Calculate the activation for this neuron
        activation = calculate_activation(neuron_weights, neuron_bias, layer_1_output)
        # The activation is directly used here (no ReLU)
        final_output.append(activation)

    # Apply softmax to the final layer output
    return softmax(final_output)

# Käytetään forward_pass-funktiota jokaiseen syötteeseen
manual_predictions = [forward_pass(sample) for sample in x_train_normal]

def manual_sparse_categorical_crossentropy(y_true, y_pred):
    total_loss = 0 # muuttujan alustus (ei ole aina 0)
    epsilon = 1e-15  #Epsilon (ε) Epsilonia käytetään tässä yhteydessä pienenä positiivisena lukuna estämään nollalla jakamisen tai laskennalliset ongelmat.

    for i in range(len(y_true)):
        # Otetaan todellinen luokka ja ennustettu todennäköisyys
        true_label = y_true[i]
        predicted_probs = y_pred[i]
        # Epsilonia käytetään estämään logaritmin argumentin nollaaminen
        true_label_prob = max(predicted_probs[true_label], epsilon)
        # Lasketaan tappio ja summaa se kokonaistappioon
        total_loss += -math.log(true_label_prob)

    # Lasketaan keskimääräinen tappio
    return total_loss / len(y_true)

def manual_accuracy(y_true, y_pred):
    correct_predictions = 0

    for i in range(len(y_true)):
        # Etsitään ennustetun luokan indeksi
        predicted_label = y_pred[i].index(max(y_pred[i]))
        # Lisätään oikein ennustettujen lukumäärää, jos ennustus on oikein
        if predicted_label == y_true[i]:
            correct_predictions += 1

    # Lasketaan tarkkuus
    return correct_predictions / len(y_true)

# Lasketaan kokonaistappio ja tarkkuus
manual_total_loss = manual_sparse_categorical_crossentropy(y_train_normal, manual_predictions)
manual_total_accuracy = manual_accuracy(y_train_normal, manual_predictions)

# Tulostetaan tappio ja tarkkuus
print(f"Manual Evaluation - Loss: {manual_total_loss}, Accuracy: {manual_total_accuracy}")

