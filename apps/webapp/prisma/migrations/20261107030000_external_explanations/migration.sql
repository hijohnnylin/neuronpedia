-- Fail fast instead of waiting behind long queries on Explanation.
SET lock_timeout = '10s';

-- CreateEnum
CREATE TYPE "ExplanationPole" AS ENUM ('POSITIVE', 'NEGATIVE', 'BOTH');

-- CreateEnum
CREATE TYPE "ExplanationAspect" AS ENUM ('ACTIVATION', 'STEERING');

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "canUploadExplanations" BOOLEAN NOT NULL DEFAULT false;

-- AlterTable
ALTER TABLE "Explanation" ADD COLUMN     "aspect" "ExplanationAspect",
ADD COLUMN     "contributorName" TEXT,
ADD COLUMN     "contributorUrl" TEXT,
ADD COLUMN     "metadata" JSONB,
ADD COLUMN     "pole" "ExplanationPole",
ADD COLUMN     "sourceUrl" TEXT,
ADD COLUMN     "steerStrength" DOUBLE PRECISION;

-- AlterTable
ALTER TABLE "ExplanationType" ADD COLUMN     "citation" TEXT,
ADD COLUMN     "license" TEXT,
ADD COLUMN     "licenseUrl" TEXT;

-- Backfill: the 328 explanations of the negative autointerp type.
UPDATE "Explanation" SET "pole" = 'NEGATIVE' WHERE "typeName" = 'np_min-act-logits';
