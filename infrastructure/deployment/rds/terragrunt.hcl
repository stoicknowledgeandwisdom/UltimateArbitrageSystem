terraform {
  source = "./"
}

inputs = {
  project_name = local.project_name
  environment  = local.environment
  region       = local.aws_region

  db_instance_class          = var.db_instance_class
  db_allocated_storage       = var.db_allocated_storage
  db_backup_retention_period = var.db_backup_retention_period
  db_multi_az                = var.db_multi_az
  db_encryption              = var.db_encryption
}

