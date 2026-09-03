-- Two changes to how a `Vector` says what it is and how it is read. The reasoning is on the models
-- in schema.prisma.
--
-- 1. `VectorTag` and `VectorTagOnVector`: free-form labels, replacing the `projectionType`
--    discriminator. A tag is a display and browsing facet -- nothing gates a read on one, because
--    every vector projects the same way and refusing a probe would withhold a well-defined number.
-- 2. `projectionParams.read`: the capture site, token selection and pooling that inference had
--    hard-coded. Written onto every existing row, so a reading stays reproducible when a fit made
--    under some other rule arrives.
--
-- ADDITIVE ON PURPOSE. `projectionType` is what the tags replace, and dropping it here would break
-- whatever code is serving while this runs: the deploy applies migrations at build time, so the old
-- revision keeps answering requests for the length of the build, and its axis query still names that
-- column in its WHERE clause. The drop is a second migration, in a later commit, once no deployed
-- revision reads it. Both files in one commit would be applied in the same build, which is the thing
-- being avoided.
--
-- REWRITES `projectionParams` IN PLACE. Run `ts-node scripts/vectors.ts dump --out vectors.json`
-- against the target database first; that file is the only record of what the rows said beforehand.
--
-- The inserts are `ON CONFLICT DO NOTHING` and the backfill skips a row that already declares a
-- `read`, so a database where some of this was done by hand takes the rest of it rather than
-- failing halfway.

-- CreateTable
CREATE TABLE "VectorTag" (
    "name" TEXT NOT NULL,
    "displayName" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "creatorId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "VectorTag_pkey" PRIMARY KEY ("name")
);

-- CreateTable
CREATE TABLE "VectorTagOnVector" (
    "vectorId" TEXT NOT NULL,
    "tagName" TEXT NOT NULL,
    "addedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "VectorTagOnVector_pkey" PRIMARY KEY ("vectorId","tagName")
);

-- CreateIndex
CREATE INDEX "VectorTagOnVector_tagName_idx" ON "VectorTagOnVector"("tagName");

-- AddForeignKey
ALTER TABLE "VectorTag" ADD CONSTRAINT "VectorTag_creatorId_fkey" FOREIGN KEY ("creatorId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VectorTagOnVector" ADD CONSTRAINT "VectorTagOnVector_vectorId_fkey" FOREIGN KEY ("vectorId") REFERENCES "Vector"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VectorTagOnVector" ADD CONSTRAINT "VectorTagOnVector_tagName_fkey" FOREIGN KEY ("tagName") REFERENCES "VectorTag"("name") ON DELETE CASCADE ON UPDATE CASCADE;

-- The tags there are, written out rather than derived: a tag is a decision, and its description is
-- what makes two people apply it the same way. No `creatorId` -- nobody claimed these.
INSERT INTO "VectorTag" ("name", "displayName", "description") VALUES
  ('axis', 'Axis',
   'A direction with two named poles. A reading is signed: toward one pole or the other, with a percentile against the conversations it was fitted on.'),
  ('probe', 'Probe',
   'A direction fitted to detect one thing. A reading says how strongly it is present, in the direction''s own units, with no opposite end implied.'),
  ('steering-vector', 'Steering vector',
   'A direction meant to be added to activations during generation, rather than read off them.')
ON CONFLICT ("name") DO NOTHING;

-- The read spec inference has been applying all along, now recorded per row. `render` stays a
-- sibling of `read`: it changes the text the model sees, so it belongs to the request rather than
-- to one vector.
UPDATE "Vector"
SET "projectionParams" = "projectionParams" || jsonb_build_object(
      'read', jsonb_build_object('site', 'resid_post', 'tokens', 'assistant_turns', 'pool', 'mean'))
WHERE NOT ("projectionParams" ? 'read');

-- Tag the rows there are. An axis is a row with both poles named that reads at one site; anything
-- else is a probe until somebody says otherwise. A wrong guess here is a wrong facet on a listing
-- page, not a reading anybody is refused.
INSERT INTO "VectorTagOnVector" ("vectorId", "tagName")
SELECT "id", 'axis' FROM "Vector"
WHERE "polePositive" IS NOT NULL AND "poleNegative" IS NOT NULL AND array_length("layers", 1) = 1
ON CONFLICT ("vectorId", "tagName") DO NOTHING;

INSERT INTO "VectorTagOnVector" ("vectorId", "tagName")
SELECT v."id", 'probe' FROM "Vector" v
WHERE NOT EXISTS (SELECT 1 FROM "VectorTagOnVector" t WHERE t."vectorId" = v."id")
ON CONFLICT ("vectorId", "tagName") DO NOTHING;
