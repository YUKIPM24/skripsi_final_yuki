# Membuka file negative.txt dan membaca isinya
# with open('negative.txt', 'r') as file:
#     lexicon = file.readlines()

with open('positive.txt', 'r') as file:
    lexicon = file.readlines()

# Menghapus karakter newline dari setiap kata
lexicon = [word.strip() for word in lexicon]

# Menambahkan kata-kata baru sesuai dengan topik skripsi Anda
# negatif
# new_words = ["malu", "karat", "kuning", "patah", "lepas tangan", "silikat", "ahmpas", "recall", "lawak", "berhentikan", 
#              "henti", "boraks", "anti", "nyawa", "rawan", "kerupuk", "burik", "berkarat", "karat", "busuk", "jangan pilih",
#              "blunder", "palaron", "stop", "lipat", "muka ditutup", "lawak", "ngelawak", "terlalu", "kasus", "cat", "terlalu ngelawak",
#              "bodo amat", "produk ampas", "ganti rangka", "ganti", "menolak", "tipis", "ngelipat", "pindah", "silit", "ngeles", "melawan",
#              "stop beli", "jual", "sakit hati", "bye", "bye bye", "kecewa", "kapok", "umur", "cemen", "kasihan", "bully", "bangkrut"
#              , "cacat", "krispi", "kaleng", "merugikan", "ganti", "rugi", "tinggalkan", "gulung", "ampas", "bobrok", "ngeri", "viralkan", "tinggalkan"]

# positif
new_words = ["makasih", "makasih min", "mantap min", "sangat bermanfaat", "top banget", "sangat bagus", 
             "thanks min", "tetap pakai honda", "percaya honda", "perbedaan", "info penting", "tenang", 
             "tenang deh", "penting", "top", "aman", "nyaman", "yakin", "pakai", "setia", "tetap setia", "menghargai", "sangat menghargai", "keren",
            "semoga", "gratis"]


# Menambahkan kata-kata baru ke lexicon yang sudah ada jika belum ada dalam kamus
for word in new_words:
    if word not in lexicon:
        lexicon.append(word)

# Susun lexicon secara alfabetis
lexicon.sort()

# Menyimpan kembali ke file negative.txt
# with open('negative.txt', 'w') as file:
#     for word in lexicon:
#         file.write(word + '\n')

with open('positive.txt', 'w') as file:
    for word in lexicon:
        file.write(word + '\n')

print("Kamus lexicon berhasil diperbarui dan disusun secara alfabetis.")
