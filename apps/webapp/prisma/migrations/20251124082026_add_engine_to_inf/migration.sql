-- CreateEnum
CREATE TYPE "InferenceEngine" AS ENUM ('TRANSFORMER_LENS', 'NNSIGHT', 'CSPACE');

-- AlterTable
ALTER TABLE "InferenceHostSource" ADD COLUMN     "engine" "InferenceEngine" NOT NULL DEFAULT 'TRANSFORMER_LENS';
