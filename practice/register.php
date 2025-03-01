<?php
// register.php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $conn = new mysqli("localhost", "root", "", "agmsdb");

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
    
    $name = mysqli_real_escape_string($conn, $_POST['name']);
    $email = mysqli_real_escape_string($conn, $_POST['email']);
    $mobile_no = mysqli_real_escape_string($conn, $_POST['mobile_no']);
    $password = $_POST['password'];
    
    $sql = "INSERT INTO user_register (name, email, mobile_no, password) VALUES ('$name', '$email', '$mobile_no', '$password')";
    
    if ($conn->query($sql) === TRUE) {
        echo "Registration successful!";
        header("Location: login.php");
    } else {
        echo "Error: " . $conn->error;
    }
    
    $conn->close();
}
?>
