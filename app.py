from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Pengaturan halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Pelajar",
    page_icon="🎓",
    layout="centered"
)


# Lokasi file model
LOKASI_MODEL = Path(__file__).parent / "model_kelulusan.pkl"


@st.cache_resource
def memuat_model():
    """Membaca model yang sebelumnya sudah dilatih di Google Colab."""
    if not LOKASI_MODEL.exists():
        raise FileNotFoundError(
            "File model_kelulusan.pkl tidak ditemukan."
        )

    return joblib.load(LOKASI_MODEL)


# Memuat model
try:
    model = memuat_model()
except Exception as error:
    st.error(f"Model gagal dimuat: {error}")
    st.stop()


# Judul aplikasi
st.title("Prediksi Kelulusan Pelajar")

st.write(
    """
    Aplikasi ini digunakan untuk memperkirakan kelulusan pelajar
    berdasarkan waktu belajar setiap minggu dan jumlah ketidakhadiran.
    """
)

st.info(
    """
    Hasil yang diberikan hanya berupa perkiraan berdasarkan pola
    dataset dan bukan keputusan kelulusan resmi.
    """
)


# Pilihan waktu belajar sesuai kode pada dataset
pilihan_waktu_belajar = {
    "Kurang dari 2 jam": 1,
    "2 sampai 5 jam": 2,
    "5 sampai 10 jam": 3,
    "Lebih dari 10 jam": 4
}


# Formulir masukan
with st.form("form_prediksi"):
    waktu_belajar_teks = st.selectbox(
        "Waktu belajar setiap minggu",
        options=list(pilihan_waktu_belajar.keys())
    )

    jumlah_tidak_hadir = st.number_input(
        "Jumlah ketidakhadiran",
        min_value=0,
        max_value=93,
        value=0,
        step=1
    )

    tombol_prediksi = st.form_submit_button(
        "Lihat Hasil Prediksi",
        use_container_width=True
    )


# Proses prediksi
if tombol_prediksi:
    waktu_belajar = pilihan_waktu_belajar[
        waktu_belajar_teks
    ]

    data_masukan = pd.DataFrame(
        [
            {
                "waktu_belajar": waktu_belajar,
                "jumlah_tidak_hadir": int(
                    jumlah_tidak_hadir
                )
            }
        ]
    )

    try:
        prediksi = int(
            model.predict(data_masukan)[0]
        )

        peluang = model.predict_proba(
            data_masukan
        )[0]

        posisi_kelas = list(
            model.classes_
        ).index(prediksi)

        tingkat_keyakinan = (
            peluang[posisi_kelas] * 100
        )

        st.divider()
        st.subheader("Hasil Prediksi")

        if prediksi == 1:
            st.success("Lulus")

            st.write(
                """
                Berdasarkan pola data yang dimasukkan,
                pelajar diperkirakan masuk ke dalam
                kelompok **Lulus**.
                """
            )
        else:
            st.warning("Belum Lulus")

            st.write(
                """
                Berdasarkan pola data yang dimasukkan,
                pelajar diperkirakan masuk ke dalam
                kelompok **Belum Lulus**.
                """
            )

        st.metric(
            label="Tingkat Keyakinan Model",
            value=f"{tingkat_keyakinan:.2f}%"
        )

        st.subheader("Data yang Dimasukkan")

        kolom_1, kolom_2 = st.columns(2)

        with kolom_1:
            st.write("**Waktu belajar**")
            st.write(waktu_belajar_teks)

        with kolom_2:
            st.write("**Ketidakhadiran**")
            st.write(
                f"{int(jumlah_tidak_hadir)} kali"
            )

    except Exception as error:
        st.error(
            f"Prediksi gagal dilakukan: {error}"
        )


st.divider()

st.caption(
    "Proyek UTS Artificial Intelligence — "
    "Deni Trio Saputra | 24110300009 | AI03"
)