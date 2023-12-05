<?php
#$text = "PTM ribet";
$text = $_POST['tweet'];
echo shell_exec("C:/Users/cahyo/AppData/Local/Programs/Python/Python310/python.exe Cek_Label.py $text");
