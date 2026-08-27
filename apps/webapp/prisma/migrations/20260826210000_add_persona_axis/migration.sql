-- PersonaAxis reached production from the np-mit fork via `prisma db push`, which
-- writes no migration file. The table therefore exists in production but in no
-- other database, and every `migrate diff` against production proposes dropping
-- it. This adopts the table into the committed history: production keeps its
-- rows, and every other database finally gets the table.
--
-- The DDL is Prisma's own, generated from the model rather than written by hand,
-- so it matches what `db push` created. Every statement is guarded, because on
-- production all of this already exists and none of it should run.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'PersonaAxisNormalize') THEN
        CREATE TYPE "PersonaAxisNormalize" AS ENUM ('L2', 'NONE');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS "PersonaAxis" (
    "id" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "author" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "retiredAt" TIMESTAMP(3),
    "layer" INTEGER NOT NULL,
    "polePositive" TEXT NOT NULL,
    "poleNegative" TEXT NOT NULL,
    "polePositiveDescription" TEXT,
    "poleNegativeDescription" TEXT,
    "displayName" TEXT,
    "caveat" TEXT,
    "direction" DOUBLE PRECISION[],
    "preNormMean" DOUBLE PRECISION[] DEFAULT ARRAY[]::DOUBLE PRECISION[],
    "postNormMean" DOUBLE PRECISION[] DEFAULT ARRAY[]::DOUBLE PRECISION[],
    "normalize" "PersonaAxisNormalize" NOT NULL DEFAULT 'NONE',
    "center" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "scalePos" DOUBLE PRECISION NOT NULL DEFAULT 1,
    "scaleNeg" DOUBLE PRECISION NOT NULL DEFAULT 1,
    "quantileLevels" DOUBLE PRECISION[] DEFAULT ARRAY[]::DOUBLE PRECISION[],
    "quantilesPos" DOUBLE PRECISION[] DEFAULT ARRAY[]::DOUBLE PRECISION[],
    "quantilesNeg" DOUBLE PRECISION[] DEFAULT ARRAY[]::DOUBLE PRECISION[],
    "blankSystemPrompt" BOOLEAN NOT NULL DEFAULT false,
    "templateKwargs" JSONB,
    "provenance" JSONB,
    "hfRepoId" TEXT,
    "hfFolderId" TEXT,
    "visibility" "Visibility" NOT NULL DEFAULT 'PRIVATE',
    "creatorId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PersonaAxis_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "PersonaAxis_modelId_visibility_idx" ON "PersonaAxis"("modelId", "visibility");

CREATE INDEX IF NOT EXISTS "PersonaAxis_creatorId_idx" ON "PersonaAxis"("creatorId");

CREATE UNIQUE INDEX IF NOT EXISTS "PersonaAxis_modelId_name_version_key" ON "PersonaAxis"("modelId", "name", "version");

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'PersonaAxis_modelId_fkey') THEN
        ALTER TABLE "PersonaAxis" ADD CONSTRAINT "PersonaAxis_modelId_fkey"
            FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'PersonaAxis_creatorId_fkey') THEN
        ALTER TABLE "PersonaAxis" ADD CONSTRAINT "PersonaAxis_creatorId_fkey"
            FOREIGN KEY ("creatorId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
