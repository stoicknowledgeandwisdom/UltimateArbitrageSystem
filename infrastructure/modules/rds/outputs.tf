output "db_instance_id" {
  description = "ID of the RDS instance"
  value       = module.rds.db_instance_id
}

output "db_instance_arn" {
  description = "ARN of the RDS instance"
  value       = module.rds.db_instance_arn
}

output "db_endpoint" {
  description = "DNS address of the RDS instance"
  value       = module.rds.db_instance_endpoint
}

output "db_host" {
  description = "The host to connect to"
  value       = module.rds.db_instance_address
}

output "db_username" {
  description = "Username for the DB connection"
  value       = module.rds.db_instance_username
}

output "db_password" {
  description = "Derived password from Secrets Manager"
  value       = random_password.password.result
}

output "db_security_group_id" {
  description = "ID of the database security group"
  value       = aws_security_group.rds.id
}

