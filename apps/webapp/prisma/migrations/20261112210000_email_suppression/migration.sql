-- CreateEnum
CREATE TYPE "EmailSuppressionReason" AS ENUM ('BOUNCE', 'COMPLAINT');

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "emailSuppressedAt" TIMESTAMP(3),
ADD COLUMN     "emailSuppressedReason" "EmailSuppressionReason";
