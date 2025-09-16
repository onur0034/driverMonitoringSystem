# DroidCam Uyku Tespit Sistemi

Bu Python projesi, DroidCam uygulaması ile yayınlanan IP kamerasını kullanarak gelişmiş uyku tespit sistemi sunar. MediaPipe kullanarak göz takibi yapar ve uyku durumunu tespit eder.

## Özellikler

- DroidCam IP kamerasından gerçek zamanlı video akışı
- MediaPipe ile gelişmiş yüz ve göz takibi
- İris takibi ile hassas göz kapanma tespiti
- Gerçek zamanlı EAR (Eye Aspect Ratio) hesaplama
- Uyku uyarı sistemi
- Modern ve kullanıcı dostu arayüz
- FPS göstergesi ve performans metrikleri

## Kurulum

### Gereksinimler

- Python 3.7 veya üzeri
- DroidCam uygulaması (Android/iOS)
- Aynı ağda bulunan cihazlar

### Adımlar

1. Projeyi klonlayın veya indirin:
```bash
git clone <repository-url>
cd opencv
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. DroidCam uygulamasını telefonunuzda açın ve IP kamerasını başlatın

## Kullanım

### Temel Kullanım

```bash
python main.py
```

Bu komut DroidCam IP kamerasına bağlanır ve uyku tespit sistemini başlatır.

### IP Adresi Değiştirme

`main.py` dosyasının 8. satırındaki URL'yi değiştirin:
```python
URL = "http://YOUR_IP_ADDRESS:4747/video"
```

## Kontroller

- **q**: Uygulamadan çık
- Sistem otomatik olarak uyku durumunu tespit eder ve uyarı verir

## DroidCam Kurulumu

1. **Android/iOS cihazınızda DroidCam uygulamasını indirin**
2. **Uygulamayı açın ve "Start Server" butonuna basın**
3. **IP adresini not edin (örn: 192.168.1.100:4747)**
4. **Bilgisayarınızda bu uygulamayı çalıştırın**

## Sorun Giderme

### Bağlantı Sorunları

- DroidCam uygulamasının çalıştığından emin olun
- IP adresinin doğru olduğunu kontrol edin
- Her iki cihazın da aynı ağda olduğunu doğrulayın
- Güvenlik duvarı ayarlarını kontrol edin

### Performans Sorunları

- Ağ bağlantınızın stabil olduğundan emin olun
- DroidCam uygulamasında video kalitesini düşürün
- Diğer ağ kullanımını azaltın

## Geliştirme

### Proje Yapısı

```
opencv/
├── main.py              # Ana uyku tespit sistemi
├── requirements.txt     # Python bağımlılıkları
└── README.md           # Bu dosya
```

### Sistem Parametreleri

`main.py` dosyasında ayarlanabilir parametreler:

- `EAR_STRICT`: Göz açıklık eşiği (varsayılan: 0.19)
- `IRIS_STRICT`: İris oran eşiği (varsayılan: 0.55)
- `CLOSED_MIN_DURATION`: Uyku tespit süresi (varsayılan: 3.0 saniye)
- `SHOW_FPS`: FPS göstergesi (varsayılan: True)

### Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.
