-- Expand-only. Creates the ComputeHost registry and backfills it from
-- InferenceHostSource, GraphHostSource and NlaSource.servers[]. The source
-- tables are left in place so a deploy can be rolled back by reverting code
-- alone; a later migration drops them.

-- CreateEnum
CREATE TYPE "ComputeService" AS ENUM ('INFERENCE', 'GRAPH', 'NLA', 'AUTOINTERP', 'SPARSITY');

-- CreateTable
CREATE TABLE "ComputeHost" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "hostUrl" TEXT NOT NULL,
    "service" "ComputeService" NOT NULL,
    "provider" TEXT,
    "providerRef" TEXT,
    "modelId" TEXT NOT NULL,
    "nlaSourceId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ComputeHost_pkey" PRIMARY KEY ("id")
);

-- An NLA process fixes its verbalizer, reconstructor and extraction layer at
-- startup, so an NLA host serves exactly one NlaSource and must name it. No
-- other service has that column set.
ALTER TABLE "ComputeHost" ADD CONSTRAINT "ComputeHost_nlaSourceId_matches_service"
    CHECK (("service" = 'NLA') = ("nlaSourceId" IS NOT NULL));

-- CreateTable
CREATE TABLE "ComputeHostOnSource" (
    "sourceId" TEXT NOT NULL,
    "sourceModelId" TEXT NOT NULL,
    "computeHostId" TEXT NOT NULL,

    CONSTRAINT "ComputeHostOnSource_pkey" PRIMARY KEY ("sourceId","sourceModelId","computeHostId")
);

-- CreateTable
CREATE TABLE "ComputeHostOnSourceSet" (
    "sourceSetName" TEXT NOT NULL,
    "sourceSetModelId" TEXT NOT NULL,
    "computeHostId" TEXT NOT NULL,

    CONSTRAINT "ComputeHostOnSourceSet_pkey" PRIMARY KEY ("sourceSetName","sourceSetModelId","computeHostId")
);

-- CreateIndex
CREATE INDEX "ComputeHost_service_modelId_idx" ON "ComputeHost"("service", "modelId");

-- CreateIndex
CREATE UNIQUE INDEX "ComputeHost_hostUrl_service_key" ON "ComputeHost"("hostUrl", "service");

-- CreateIndex
CREATE INDEX "ComputeHostOnSource_sourceModelId_idx" ON "ComputeHostOnSource"("sourceModelId");

-- CreateIndex
CREATE INDEX "ComputeHostOnSource_sourceModelId_sourceId_idx" ON "ComputeHostOnSource"("sourceModelId", "sourceId");

-- CreateIndex
CREATE INDEX "ComputeHostOnSourceSet_sourceSetModelId_idx" ON "ComputeHostOnSourceSet"("sourceSetModelId");

-- CreateIndex
CREATE INDEX "ComputeHostOnSourceSet_sourceSetModelId_sourceSetName_idx" ON "ComputeHostOnSourceSet"("sourceSetModelId", "sourceSetName");

-- AddForeignKey
ALTER TABLE "ComputeHost" ADD CONSTRAINT "ComputeHost_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ComputeHostOnSource" ADD CONSTRAINT "ComputeHostOnSource_sourceId_sourceModelId_fkey" FOREIGN KEY ("sourceId", "sourceModelId") REFERENCES "Source"("id", "modelId") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ComputeHostOnSource" ADD CONSTRAINT "ComputeHostOnSource_computeHostId_fkey" FOREIGN KEY ("computeHostId") REFERENCES "ComputeHost"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ComputeHostOnSourceSet" ADD CONSTRAINT "ComputeHostOnSourceSet_sourceSetName_sourceSetModelId_fkey" FOREIGN KEY ("sourceSetName", "sourceSetModelId") REFERENCES "SourceSet"("name", "modelId") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ComputeHostOnSourceSet" ADD CONSTRAINT "ComputeHostOnSourceSet_computeHostId_fkey" FOREIGN KEY ("computeHostId") REFERENCES "ComputeHost"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ComputeHost" ADD CONSTRAINT "ComputeHost_modelId_nlaSourceId_fkey" FOREIGN KEY ("modelId", "nlaSourceId") REFERENCES "NlaSource"("modelId", "id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Backfill.
--
-- Old ids are reused as ComputeHost ids where possible. Rows are deduplicated
-- on hostUrl, since the engine column allowed several rows to share one URL and
-- (hostUrl, service) is now unique. The join backfills below therefore match on
-- hostUrl rather than on the old id, so links survive that collapse.

INSERT INTO "ComputeHost" ("id", "name", "hostUrl", "service", "modelId", "provider", "createdAt", "updatedAt")
SELECT DISTINCT ON (ihs."hostUrl")
    ihs."id", ihs."name", ihs."hostUrl", 'INFERENCE'::"ComputeService",
    ihs."modelId", 'runpod', ihs."createdAt", ihs."updatedAt"
FROM "InferenceHostSource" ihs
ORDER BY ihs."hostUrl", ihs."createdAt"
ON CONFLICT ("hostUrl", "service") DO NOTHING;

INSERT INTO "ComputeHost" ("id", "name", "hostUrl", "service", "modelId", "provider", "createdAt", "updatedAt")
SELECT DISTINCT ON (ghs."hostUrl")
    ghs."id", ghs."name", ghs."hostUrl", 'GRAPH'::"ComputeService",
    ghs."modelId", 'runpod', ghs."createdAt", ghs."updatedAt"
FROM "GraphHostSource" ghs
WHERE ghs."hostUrl" IS NOT NULL AND ghs."hostUrl" <> ''
ORDER BY ghs."hostUrl", ghs."createdAt"
ON CONFLICT ("hostUrl", "service") DO NOTHING;

-- One NLA server serves one NlaSource, so a URL listed under two sources was
-- already misrouting half its traffic. Say so rather than silently picking one.
DO $$
DECLARE conflicting TEXT;
BEGIN
    SELECT string_agg(srv, ', ') INTO conflicting
    FROM (
        SELECT srv FROM "NlaSource" ns, unnest(ns."servers") AS srv
        WHERE srv <> '' GROUP BY srv HAVING count(DISTINCT ns."id") > 1
    ) dupes;
    IF conflicting IS NOT NULL THEN
        RAISE WARNING 'NLA server(s) listed under more than one NlaSource: %. Keeping the oldest source for each; re-register the rest.', conflicting;
    END IF;
END $$;

INSERT INTO "ComputeHost" ("id", "name", "hostUrl", "service", "modelId", "nlaSourceId", "provider", "createdAt", "updatedAt")
SELECT DISTINCT ON (srv)
    gen_random_uuid()::TEXT, ns."id", srv, 'NLA'::"ComputeService",
    ns."modelId", ns."id", 'runpod', ns."createdAt", CURRENT_TIMESTAMP
FROM "NlaSource" ns, unnest(ns."servers") AS srv
WHERE srv <> ''
ORDER BY srv, ns."createdAt"
ON CONFLICT ("hostUrl", "service") DO NOTHING;

INSERT INTO "ComputeHostOnSource" ("sourceId", "sourceModelId", "computeHostId")
SELECT DISTINCT ihsos."sourceId", ihsos."sourceModelId", ch."id"
FROM "InferenceHostSourceOnSource" ihsos
JOIN "InferenceHostSource" ihs ON ihs."id" = ihsos."inferenceHostId"
JOIN "ComputeHost" ch ON ch."hostUrl" = ihs."hostUrl" AND ch."service" = 'INFERENCE'
ON CONFLICT DO NOTHING;

INSERT INTO "ComputeHostOnSourceSet" ("sourceSetName", "sourceSetModelId", "computeHostId")
SELECT DISTINCT ghsoss."sourceSetName", ghsoss."sourceSetModelId", ch."id"
FROM "GraphHostSourceOnSourceSet" ghsoss
JOIN "GraphHostSource" ghs ON ghs."id" = ghsoss."graphHostSourceId"
JOIN "ComputeHost" ch ON ch."hostUrl" = ghs."hostUrl" AND ch."service" = 'GRAPH'
ON CONFLICT DO NOTHING;

