# ─────────────────────────────────────────────
# main.tf — Phase 5: AWS EC2 Deployment
# Spaceship Titanic MLOps Pipeline
# ─────────────────────────────────────────────

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.3.0"
}

# ── Provider ───────────────────────────────────
provider "aws" {
  region = var.aws_region
}

# ── Data: Latest Ubuntu 22.04 AMI ──────────────
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── Security Group ─────────────────────────────
resource "aws_security_group" "spaceship_sg" {
  name        = "${var.app_name}-sg"
  description = "Security group for Spaceship Titanic MLOps app"

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # FastAPI
  ingress {
    description = "FastAPI"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # MLflow UI
  ingress {
    description = "MLflow UI"
    from_port   = 5001
    to_port     = 5001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.app_name}-sg"
    Project = var.app_name
  }
}

# ── EC2 Instance ───────────────────────────────
resource "aws_instance" "spaceship_app" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.spaceship_sg.id]

  # User data — runs on first boot
  user_data = templatefile("${path.module}/user_data.sh", {
    github_repo   = var.github_repo_url
    repo_branch   = var.repo_branch
    app_name      = var.app_name
  })

  root_block_device {
    volume_size = 20    # GB — enough for Docker + model
    volume_type = "gp3"
  }

  tags = {
    Name    = var.app_name
    Project = var.app_name
    Env     = "dev"
  }
}

# ── Elastic IP (stable public IP) ─────────────
resource "aws_eip" "spaceship_eip" {
  instance = aws_instance.spaceship_app.id
  domain   = "vpc"

  tags = {
    Name    = "${var.app_name}-eip"
    Project = var.app_name
  }
}
