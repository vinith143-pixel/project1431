<?php
// login_deatils.php
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['login'])) {
    $conn = new mysqli("localhost", "root", "", "agmsdb");

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
    
    if (empty($_POST['email']) || empty($_POST['password'])) {
        die("Email or password is missing!");
    }

    $email = mysqli_real_escape_string($conn, $_POST['email']);
    $password = $_POST['password'];
    
    $sql = "SELECT password FROM user_register WHERE email='$email'";
    $result = $conn->query($sql);
    
    if (!$result) {
        die("Query failed: " . $conn->error);
    }

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        if ($password === $row['password']) {
            echo "Login successful!";
            header("Location: index.php");
            exit();
        } else {
            echo "Invalid credentials.";
        }
    } else {
        echo "User not found.";
    }
    
    $conn->close();
}
?>
