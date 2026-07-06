-- AlterTable
ALTER TABLE "JlensShare" ADD COLUMN     "steerLayers" INTEGER[] DEFAULT ARRAY[]::INTEGER[],
ADD COLUMN     "steerStrength" DOUBLE PRECISION,
ADD COLUMN     "steerToken" TEXT,
ADD COLUMN     "steerType" TEXT;
