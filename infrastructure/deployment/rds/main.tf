provider "aws" {
  region = var.region
}

# Database Instance
module "rds" {
  source              = "terraform-aws-modules/rds/aws"
  engine              = "postgres"
  engine_version      = "13.5"
  instance_class      = var.db_instance_class
  allocated_storage   = var.db_allocated_storage
  name                = "${var.project_name}-${var.environment}"
  username            = "admin"
  password            = local.db_password
  subnet_ids          = data.aws_subnets.private.ids
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Multi-AZ
  multi_az = var.db_multi_az

  # Storage Encryption
  storage_encrypted = var.db_encryption

  # Backup
  backup_retention_period = var.db_backup_retention_period

  tags = var.common_tags
}

# Generate random password
resource "random_password" "password" {
  length  = 16
  special = true
}

resource "aws_secretsmanager_secret" "rds" {
  name = "${var.project_name}-${var.environment}-rds-password"
}

resource "aws_secretsmanager_secret_version" "rds" {
  secret_id     = aws_secretsmanager_secret.rds.id
  secret_string = random_password.password.result
}

# Security group for RDS
resource "aws_security_group" "rds" {
  name        = "${var.project_name}-${var.environment}-rds"
  description = "Allow traffic to RDS"
  vpc_id      = data.aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-${var.environment}-rds-sg"
  })

  dynamic "ingress" {
    for_each = data.aws_subnets.private.ids
    content {
      description = "Postgres access ${ingress.key}"
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = [data.aws_vpc.main.cidr_block]
    }
  }
}

# Data sources
data "aws_vpc" "main" {
  filter {
    name   = "tag:Name"
    values = ["${var.project_name}-${var.environment}-vpc"]
  }
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }

  filter {
    name   = "tag:Type"
    values = ["private"]
  }
}

