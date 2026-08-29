-- Replace `PersonaAxis` with `Vector`: one table for any stored direction, with the
-- calibration that was twelve columns folded into a `projectionParams` blob, `layer`
-- widened to a `layers` array, and a `projectionType` tag saying which shape that blob
-- holds. The reasoning is on the model in schema.prisma.
--
-- DESTRUCTIVE, AND THE ROWS ARE NOT CARRIED HERE. `DROP TABLE "PersonaAxis"` is the
-- only copy of those axes gone. Run `scripts/axes.ts dump` against the target database
-- first, and `scripts/axes.ts load` after; see the header of that script for the full
-- sequence and why the transfer sits outside the migration.
--
-- Unguarded, unlike the migration that adopted `PersonaAxis`: that one ran against a
-- production database where the table already existed, whereas by this point in the
-- history every database has it.

-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "vector";

-- CreateEnum
CREATE TYPE "ProjectionType" AS ENUM ('AXIS_PROJECTION');

-- DropForeignKey
ALTER TABLE "PersonaAxis" DROP CONSTRAINT "PersonaAxis_creatorId_fkey";

-- DropForeignKey
ALTER TABLE "PersonaAxis" DROP CONSTRAINT "PersonaAxis_modelId_fkey";

-- DropTable
DROP TABLE "PersonaAxis";

-- DropEnum
DROP TYPE "PersonaAxisNormalize";

-- CreateTable
CREATE TABLE "Vector" (
    "id" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "author" TEXT NOT NULL,
    "layers" INTEGER[],
    "polePositive" TEXT,
    "poleNegative" TEXT,
    "polePositiveDescription" TEXT,
    "poleNegativeDescription" TEXT,
    "displayName" TEXT,
    "caveat" TEXT,
    "values" DOUBLE PRECISION[],
    "projectionType" "ProjectionType" NOT NULL DEFAULT 'AXIS_PROJECTION',
    "projectionParams" JSONB NOT NULL DEFAULT '{}',
    "hfRepoId" TEXT,
    "hfFolderId" TEXT,
    "visibility" "Visibility" NOT NULL DEFAULT 'PRIVATE',
    "creatorId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Vector_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Vector_modelId_visibility_idx" ON "Vector"("modelId", "visibility");

-- CreateIndex
CREATE INDEX "Vector_creatorId_idx" ON "Vector"("creatorId");

-- CreateIndex
CREATE UNIQUE INDEX "Vector_modelId_name_key" ON "Vector"("modelId", "name");

-- AddForeignKey
ALTER TABLE "Vector" ADD CONSTRAINT "Vector_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Vector" ADD CONSTRAINT "Vector_creatorId_fkey" FOREIGN KEY ("creatorId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
