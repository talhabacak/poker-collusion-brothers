# Detect Suspicious Value Transfers in Poker

Kaggle yarışması: tamamen sentetik No-Limit Texas Hold'em (NLHE) verisinde **birlikte hareket eden oyuncu çiftlerini** bulmak, davranış türlerini belirlemek ve her alarm için bir incelemecinin açıp bakabileceği **somut elleri** kanıt olarak göstermek.

> Veri tamamen sentetiktir. Gerçek oyuncu, müşteri, ödeme ya da üretim verisi içermez.

---

## 1. Problem ne?

Şüpheli görünen oyunun çoğu zaman masum bir açıklaması olur: tilt, tecrübesizlik, aynı saatlerde oynamak, sıra dışı bir strateji ya da düz şans. Anlaşmalı oyuncular ise hileyi sürekli yapmaz. Manipüle ettikleri elleri normal oyunun arasına serpiştirirler.

Bizden istenen sistemin üç şey yapması:

1. **Çift sıralama:** Her oyuncu çifti için `0–1` arası bir risk skoru üretmek.
2. **Davranış sınıflandırma:** Şüpheli çiftin hangi koordinasyon türüne girdiğini söylemek.
3. **Kanıt getirme:** Koordinasyonun gözle görülebildiği en güçlü 5 eli, güçlüden zayıfa sıralı olarak vermek.

Sadece yüksek risk skoru yetmiyor. Skorun bir kısmı doğru elleri bulmaktan geliyor, bu yüzden "bu iki oyuncu çok birlikte oynuyor" ya da "aralarında çok çip akmış" gibi kaba sinyallerle sınırlı kalmak yeterli olmayacak.

---

## 2. Hedef davranışlar

| Sınıf (`predicted_behavior`) | Anlamı | Aksiyon logunda olası iz |
|---|---|---|
| `directed_transfer` | Oyunculardan biri değeri **bilerek** diğerine kaybeder (chip dumping). | Zayıf elle büyük bet/call, ortağa karşı çok iyi eli fold'lamak, ortak lehine anlamsız all-in. |
| `soft_play` | Ortaklar birbirine karşı **normal agresyonu göstermez**. | Güçlü elle ortağa karşı sadece check/call, raise yerine limp, heads-up'ta pasiflik. |
| `coordinated_isolation` | Ortaklar **diğer oyuncuları sıkıştırır**, kendi aralarında çatışmayı sınırlar. | Squeeze/re-raise ile üçüncü oyuncuyu pottan atmak, ardından ortaklar arasında pasif oyun. |
| `other_coordination` | Açıklanmayan **dördüncü bir mekanizma**. Public pozitif etiketlerde **hiç yer almaz**. | Bilinmiyor. Koordinasyon var ama yukarıdaki üç kalıba uymuyorsa kullanılır. |
| `none` | Hedef davranış yok. | — |

"Olası iz" sütunu yarışmanın resmi tanımı değil, bizim yorumumuz. Resmi olarak bilinen tek şey şu: etiketli her kanıt elinde `actions.parquet` içinde **görünür, davranışa özgü bir aksiyon** bulunuyor. Senaryonun arka planda aktif olması tek başına kanıt sayılmıyor.

### Koordinasyon epizodik

- Anlaşmalı bir çift, manipüle ettiği elleri sıradan ellerle karıştırır.
- İlişki tüm zaman çizelgesi boyunca aktif olmak zorunda değildir. Development döneminde aktif olup evaluation döneminde bitmiş olabilir, ya da tam tersi.
- Bu yüzden tüm eller üzerinden alınan ortalama istatistikler sinyali boğabilir. **Pencere bazlı / el bazlı** analiz önemli.

### Tuzak örüntüler (hedef değil)

Veride bilerek konmuş, hedef davranışa benzeyen ama hedef olmayan örüntüler var:

- **Tilt:** Kaybettikten sonra agresif ya da kötü oynamak.
- **Zayıf oyun:** Genel olarak kötü oynayıp herkese çip kaybetmek.
- **Benzer stratejiler:** İki oyuncunun aynı tarzda oynaması.
- **Tekrarlanan rakip seçimi:** Hep aynı kişilerle aynı masada bulunmak.
- **Seriler (streaks):** Şans kaynaklı kazanma/kaybetme serileri.
- **Strateji değişimleri:** Oyuncunun zamanla tarz değiştirmesi.

Bunlara ek olarak oyuncular farklı zamanlarda ve farklı miktarlarda oynuyor. Bazı çiftler diğerlerinden çok daha fazla el paylaşıyor, dolayısıyla ham sayılar yerine **paylaşılan el sayısına göre normalize edilmiş** metrikler gerekiyor.

---

## 3. Veri

### Genel yapı

| Büyüklük | Değer |
|---|---|
| El (hand) | 2,000,000 (6 kişilik NLHE) |
| Oyuncu | 12,000 |
| Oyuncu-el satırı | 12,000,000 |
| Aksiyon | 18,609,028 |
| Değerlendirilecek çift | 112,540 |
| Havuz (pool) / masa | 400 havuz × 30 oyuncu, her havuz tek bir `table_id` |
| Havuz başına el | ~5,000 |
| Zaman aralığı | 2026-01-01 → 2026-02-02 (UTC) |

**Zaman bölünmesi:** Her havuz içinde ellerin ilk %60'ı `development`, son %40'ı `evaluation` dönemi (`hands.phase`).

- `development`: 1,200,000 el
- `evaluation`: 800,000 el

Oyuncular sadece kendi havuzları içinde karşılaşıyor. Olası çift sayısı havuz başına C(30,2) = 435, toplamda 174,000. Bunların 112,540'ı değerlendirmeye giriyor.

### Klasör yapısı

```
detect-suspicious-value-transfers-in-poker/
├── README.md                     ← bu dosya
└── data/raw/
    ├── players.parquet           (~137 KB)
    ├── hands.parquet             (~48 MB)
    ├── seats.parquet             (~95 MB)
    ├── actions.parquet           (~151 MB)
    ├── development_labels.csv
    ├── development_evidence.csv
    ├── evaluation_pairs.csv
    └── sample_submission.csv
```

Join anahtarları: oyun tabloları `hand_id` üzerinden, oyuncu bilgisi `player_id` üzerinden.

### `players.parquet`: 12,000 satır

| Kolon | Tip | Açıklama / değerler |
|---|---|---|
| `player_id` | str | Oyuncu kimliği |
| `account_age_days` | int | Hesap yaşı (gün) |
| `experience_hands_bucket` | str | `new` (1,411), `developing` (3,299), `experienced` (4,588), `veteran` (2,702) |
| `preferred_stake` | str | `micro` (6,595), `low` (4,198), `mid` (1,207) |
| `region_bucket` | str | `americas`, `europe`, `apac`, `other` |
| `client_family` | str | `desktop`, `mobile`, `web` |

### `hands.parquet`: 2,000,000 satır

| Kolon | Tip | Açıklama |
|---|---|---|
| `hand_id` | str | El kimliği |
| `table_id` | str | Masa (= havuz), 400 farklı değer |
| `started_at` | timestamp (UTC) | Elin başlama zamanı |
| `phase` | str | `development` / `evaluation` |
| `button_seat` | int | Dealer butonunun koltuğu |
| `small_blind`, `big_blind` | int | Blind'lar. BB değerleri: 2 (1.09M el), 4 (715K), 10 (195K) |
| `board_cards` | str | Açılan board kartları (ör. `7h Th 7d Kc`). El erken biterse kısa ya da boş olur |
| `final_pot` | int | Son pot büyüklüğü |
| `players_dealt` | int | Kart dağıtılan oyuncu sayısı |
| `players_at_showdown` | int | Showdown'a kalan oyuncu sayısı |

### `seats.parquet`: 12,000,000 satır (el başına 6 oyuncu)

| Kolon | Tip | Açıklama |
|---|---|---|
| `hand_id`, `player_id` | str | Anahtarlar |
| `seat_no` | int | Koltuk numarası |
| `starting_stack` | int | Elin başındaki stack |
| `hole_card_1`, `hole_card_2` | str | Oyuncunun kapalı kartları. **Fold eden dahil herkes için var**, bu da el gücü analizini mümkün kılıyor |
| `total_contribution` | int | Pota koyulan toplam çip |
| `net_chips` | int | Elden net kazanç/kayıp |
| `folded` | bool | Fold etti mi |
| `went_to_showdown` | bool | Showdown'a gitti mi |
| `won_share` | float | Kazanılan pot payı (split pot için 0–1) |

### `actions.parquet`: 18,609,028 satır

| Kolon | Tip | Açıklama |
|---|---|---|
| `hand_id` | str | El |
| `action_no` | int | El içindeki aksiyon sırası (0'dan başlar) |
| `street` | str | `preflop` (13.2M), `flop` (2.85M), `turn` (1.61M), `river` (0.91M) |
| `player_id` | str | Aksiyonu yapan |
| `action` | str | `fold` (9.65M), `call` (2.97M), `raise` (2.47M), `check` (1.84M), `bet` (1.50M), `all_in` (176K) |
| `amount` | int | Bu aksiyonda koyulan miktar |
| `amount_to` | int | Aksiyon sonrası toplam bet seviyesi |
| `pot_before` | int | Aksiyon öncesi pot |
| `stack_before` | int | Aksiyon öncesi stack |
| `to_call` | int | Karar anında call için gereken miktar |
| `players_active` | int | Karar anında elde kalan oyuncu sayısı |

Karar anı bağlamı (`pot_before`, `to_call`, `stack_before`, `players_active`) ile kapalı kartlar birlikte kullanılınca pot odds'a ve el gücüne göre **"bu aksiyon mantıklı mıydı?"** sorusu el bazında sorulabiliyor.

### `development_labels.csv`: 1,860 satır

| Kolon | Açıklama |
|---|---|
| `pair_id` | Çift kimliği |
| `player_1`, `player_2` | Oyuncular |
| `label` | `1` = hedef, `0` = hedef değil |
| `label_status` | `confirmed_target` (372) / `confirmed_non_target` (1,488) |
| `behavior_family` | `directed_transfer` (148), `soft_play` (132), `coordinated_isolation` (92), `none` (1,488) |

⚠️ **Pozitif-etiketsiz (PU) öğrenme durumu:** Bu dosya development dönemindeki koordineli çiftlerin **hepsini listelemiyor**. Listede olmayan bir çift **negatif değil, bilinmiyor**. Listede olmayan çiftleri negatif kabul edip eğitim yapmak modeli yanıltır.

`confirmed_non_target` çiftleri muhtemelen tuzak örüntüleri (tilt, benzer strateji vb.) içeren "zor negatifler". Bu yüzden negatif sınıfı öğrenmek için çok değerliler.

### `development_evidence.csv`: 1,817 satır

| Kolon | Açıklama |
|---|---|
| `pair_id` | Public pozitif çift |
| `evidence_rank` | 1–5, güçlüden zayıfa |
| `hand_id` | Kanıt eli (development dönemi) |
| `behavior_family` | Davranış türü |

372 pozitif çiftin 340'ında 5, 21'inde 4, 11'inde 3 kanıt eli var. Bu eller, "kanıt eli neye benzer?" sorusuna cevap veren tek gözetimli sinyal. El seviyesinde bir **evidence scorer** eğitmek için kullanılabilir.

### `evaluation_pairs.csv`: 112,540 satır

| Kolon | Açıklama |
|---|---|
| `pair_id` | Skorlanacak çift |
| `player_1`, `player_2` | Oyuncular |
| `shared_hands` | Evaluation döneminde birlikte oynanan el sayısı. Min 38, medyan 76, ortalama ~86, max 419 |

Public olarak etiketlenmiş çift ID'leri ve **public pozitif bir oyuncu içeren** çiftler bu listeden çıkarılmış. Yani development'ta pozitif olan bir oyuncu evaluation çiftlerinde hiç yer almıyor.

---

## 4. Submission formatı

Dosya adı `submission.csv` olmalı ve `evaluation_pairs.csv` içindeki **her `pair_id` için bir satır** içermeli. Şablon olarak `sample_submission.csv` kullanılır.

| Kolon | Açıklama |
|---|---|
| `pair_id` | Değiştirmeden kopyalanır |
| `risk_score` | 0–1 arası. Yüksek değer = koordinasyon ihtimali yüksek |
| `predicted_behavior` | `none`, `directed_transfer`, `soft_play`, `coordinated_isolation`, `other_coordination` |
| `evidence_hand_1` … `evidence_hand_5` | Evaluation dönemine ait, güçlüden zayıfa sıralı `hand_id`'ler. Kullanılmayan pozisyona `NO_EVIDENCE` yazılır |

Örnek satır:

```csv
pair_id,risk_score,predicted_behavior,evidence_hand_1,evidence_hand_2,evidence_hand_3,evidence_hand_4,evidence_hand_5
P00005AC2A509,0.87,soft_play,H2D8EAC9EC7DA02,HF67EF16BB8EF76,HD9B5F56C491E2D,NO_EVIDENCE,NO_EVIDENCE
```

**Kurallar:**

- Boş hücre olmamalı. Kaggle kabul etmiyor.
- Aynı satırda aynı `hand_id` **tekrar edilirse submission geçersiz olur**.
- Kanıt elinde çiftin **iki oyuncusu da** bulunmalı.
- Bilinmeyen, ortak olmayan, development dönemine ait ya da ilgisiz eller hata vermez ama **kanıt puanı almaz**.

---

## 5. Değerlendirme

Leaderboard skoru üç bileşenden oluşuyor:

### 5.1 Pair AP

`risk_score` ile tüm evaluation çiftleri üzerinden hesaplanan Average Precision:

$$\text{AP} = \sum_n (R_n - R_{n-1})\, P_n$$

$P_n$ ve $R_n$, n'inci eşikteki precision ve recall. Eşit risk skorları `pair_id`'ye göre deterministik olarak sıralanıyor. Bu yüzden ties'tan kaçınmak iyi olur.

### 5.2 Evidence MAP@5

- Sadece **gerçek hedef çiftler** üzerinden ortalama alınıyor.
- Her hedef çift için gönderilen sıralı 5 elin, gizli "planted evidence" elleriyle karşılaştırılmasıyla AP@5 hesaplanıyor.
- Kaçırılan (kanıt verilmeyen) hedef çift **0** katkı yapıyor.
- `NO_EVIDENCE` yok sayılıyor.

### 5.3 Behavior MAP

- Açıklanmış **üç** aile (`directed_transfer`, `soft_play`, `coordinated_isolation`) için one-vs-rest AP'nin makro ortalaması.
- Bir aile için sınıf skoru şöyle: o aile tahmin edildiyse `risk_score`, edilmediyse `0`.
- Hiç tahmin edilmeyen sınıf **0** katkı yapıyor. Ortalama her zaman üç aile üzerinden alındığı için bir sınıfı hiç tahmin etmemek pahalıya patlar.
- `other_coordination` bu bileşene dahil değil.

> Bileşenlerin nihai skorda nasıl birleştirildiği (eşit ağırlıklı ortalama mı, farklı ağırlıklar mı) açıklamada belirtilmemiş. Resmi metric notebook'undan teyit edilmeli.

### Leaderboard ayrımı

- Public LB: evaluation çiftlerinin ~%30'u
- Private LB: ~%70'i
- Ayrım davranış ailesine göre stratified. Etiketler ve LB ataması gizli, metric kodu public.

### Metrikten çıkan pratik sonuçlar

- **Kanıt vermenin maliyeti yok.** Evidence MAP@5 sadece gerçek hedef çiftler üzerinden hesaplandığı için negatif çiftlere kanıt eli yazmak ceza almıyor. Düşük riskli çiftler dahil **her satıra 5 aday el** yazmak mantıklı.
- **Kanıt sıralaması önemli.** MAP@5 sıraya duyarlı, en güçlü kanıt 1. pozisyonda olmalı.
- **Kanıt eli her zaman ortak el olmalı.** Adaylar sadece iki oyuncunun da oturduğu evaluation ellerinden seçilmeli.
- **`other_coordination` riskli bir etiket.** Bu tahmin üç ailenin hiçbirine skor vermediği için çift gerçekte bilinen üç aileden biriyse Behavior MAP'te puan kaybedilir. Pair AP ve Evidence bileşenleri ise etkilenmez. Sadece üç kalıba gerçekten uymayan güçlü sinyallerde kullanılmalı.
- **Sınıf kararı ile risk skoru birbirine bağlı.** Behavior MAP'te sınıf skoru `risk_score` olduğu için yanlış sınıflandırılan yüksek riskli bir çift, hem doğru ailesinde kayıp hem yanlış ailesinde false positive yaratır.

---

## 6. Kurallar ve kazanan doğrulaması

**Yasak:** Koordinasyon **poker aktivitesinden** çıkarılmalı. Şunları kullanmak yasak:

- ID formatları (ör. `pair_id`/`player_id` içindeki desenler)
- Satır/dosya sıralaması
- Generator'ın iç yapısı
- Oyunla ilgisi olmayan her türlü artifact

**Ödüle hak kazanmak için:** Private LB açıklandıktan sonraki **7 gün içinde**, sıralamadan bağımsız olarak şunlar yayınlanmalı:

1. En fazla **1,500 kelimelik** bir Kaggle Solution Writeup.
2. Seçilen submission'ı yeniden üretebilecek kodu içeren **public notebook veya repo**.
3. Gönderilen kanıtlardan **5 kısa vaka incelemesi**. Her birinde `pair_id`, hand ID'ler, gözlemlenen davranış ve **makul bir masum alternatif açıklama** yer almalı.

Organizatör kodu temiz bir ortamda çalıştırabilir. Yeniden üretilemeyen ya da kurallara uymayan submission ödül alamaz.

Bu yüzden pipeline **baştan itibaren tekrarlanabilir** olmalı: sabit seed'ler, sabit bağımlılık versiyonları ve açık bir çalıştırma talimatı.

---

## 7. Yaklaşım için ilk notlar

Bunlar kesinleşmiş bir plan değil, başlangıç fikirleri:

1. **El seviyesi sinyaller (kanıt motoru):** Çiftin ortak olduğu her el için, iki oyuncunun birbirine karşı yaptığı aksiyonların el gücüne ve pot odds'a göre ne kadar "anormal" olduğunu ölçen skorlar üretmek. Örnekler: güçlü elle ortağa karşı pasiflik, zayıf elle ortağa büyük para koymak, üçüncü oyuncuya karşı ortak baskı. Development kanıt elleri bu scorer'ı eğitmek ya da kalibre etmek için kullanılabilir.
2. **Oyuncu baseline'ı:** Bir oyuncunun ortağa karşı davranışını, **aynı oyuncunun diğer rakiplere karşı** davranışıyla kıyaslamak. Bu, benzer strateji ve zayıf oyun gibi tuzakları eler.
3. **Çift seviyesi agregasyon:** El skorlarını çift bazında toplamak. Epizodik yapı nedeniyle ortalama yerine top-k, zaman pencereleri, yoğunluk ya da change-point istatistikleri kullanmak. Sonuçları `shared_hands`'e göre normalize etmek.
4. **PU learning:** Etiketsiz çiftleri negatif saymamak. `confirmed_non_target` çiftleri zor negatif olarak, `confirmed_target` çiftleri pozitif olarak kullanıp etiketsizleri ayrı ele almak.
5. **Doğrulama:** Development döneminde, public pozitif oyuncuları içeren çiftleri hariç tutarak evaluation koşullarını taklit eden bir CV kurmak. Yarışmanın üç metriğinin yerel bir kopyasını yazmak (resmi metric notebook'u referans alınarak).
6. **Anomali tespiti ile `other_coordination`:** Bilinen üç kalıba uymayan ama çift bazında belirgin şekilde anormal olan ilişkiler için ayrı bir dal.

---

## 8. Terimler

| Terim | Anlamı |
|---|---|
| **NLHE** | No-Limit Texas Hold'em |
| **BB** | Big blind |
| **AP** | Average Precision (çift sıralaması için) |
| **MAP@5** | En fazla 5 elden oluşan kanıt sıralaması için Mean Average Precision |
| **PU** | Positive-Unlabelled learning: sadece bazı pozitiflerin etiketli olduğu, geri kalanının bilinmediği öğrenme |
| **Chip dumping** | Bir oyuncunun bilerek başka bir oyuncuya çip kaybetmesi (`directed_transfer`) |
| **Soft play** | Anlaşmalı oyuncuların birbirine karşı agresif oynamaması |
| **Squeeze / isolation** | Rakipleri pottan atmak için yapılan agresif raise'ler |
| **Tilt** | Duygusal nedenlerle, genelde kayıp sonrası, bozulan oyun |
| **Showdown** | Son bet turundan sonra kartların açılması |
| **Development / Evaluation** | Her havuzdaki ellerin ilk %60'ı / son %40'ı |
