# Define alphabets with uppercase English letters (A-Z), digits (0-9), and special characters
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*^_)(- .',"
max_str_len = 34  # max length of input labels
num_of_characters = len(alphabets) + 1  # +1 for CTC pseudo blank
num_of_timestamps = 100  # max length of predicted labels

# Function to convert label (string) to numerical representation

def label_to_num(label):
    label_num = []
    for ch in label:
        idx = alphabets.find(ch)
        if idx == -1:  # This means the character is not found in the alphabet
            raise ValueError(f"Character '{ch}' not in alphabet.")
        label_num.append(idx)
    return np.array(label_num)


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

def decode_ctc (preds,input_length):
    decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(preds, input_length=input_length, greedy=True)[0][0])
    return decode