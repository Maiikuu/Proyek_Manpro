<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $wilayah = $_POST['WILAYAH'];
    $kecamatan = $_POST['KECAMATAN'];
    $faskes = $_POST['NAMA FASKES (Rumah Sakit dan Puskesmas)'];
    $penyakit = $_POST['Jenis Penyakit'];
    $februari = $_POST['Februari'];
    $latitude = $_POST['latitude'];
    $longitude = $_POST['longitude'];

    $file = 'C:/Codingan/Manpro/Projek/Proyek_Manpro/new_februari.csv';
    
    // Check if the file exists and is writable
    if (!file_exists($file) || !is_writable($file)) {
        echo "File not found or not writable!";
        exit();
    }

    // Prepare data to append
    $newData = "$wilayah,$kecamatan,$faskes,$penyakit,$februari,$latitude,$longitude\n";

    // Append new data to the file
    if (file_put_contents($file, $newData, FILE_APPEND) === false) {
        echo "Failed to write to file!";
        exit();
    }

    echo "Data telah berhasil diperbarui!";
} else {
    echo "Tidak ada data yang dikirim.";
}
?>
