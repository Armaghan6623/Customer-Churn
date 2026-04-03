resource "aws_db_instance" "churn_db" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "13"
  instance_class       = "db.t3.micro"
  name                 = "customer_churn_db"
  username             = "dbadmin"
  password             = "yoursecurepassword" # Change this!
  skip_final_snapshot  = true
  publicly_accessible  = true
}
