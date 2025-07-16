import { z } from 'zod';
import { Prisma } from '@prisma/client';

/////////////////////////////////////////
// HELPER FUNCTIONS
/////////////////////////////////////////

// JSON
//------------------------------------------------------

export type NullableJsonInput = Prisma.JsonValue | null | 'JsonNull' | 'DbNull' | Prisma.NullTypes.DbNull | Prisma.NullTypes.JsonNull;

export const transformJsonNull = (v?: NullableJsonInput) => {
  if (!v || v === 'DbNull') return Prisma.DbNull;
  if (v === 'JsonNull') return Prisma.JsonNull;
  return v;
};

export const JsonValue: z.ZodType<Prisma.JsonValue> = z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.lazy(() => z.array(JsonValue)),
  z.lazy(() => z.record(JsonValue)),
]);

export type JsonValueType = z.infer<typeof JsonValue>;

export const NullableJsonValue = z
  .union([JsonValue, z.literal('DbNull'), z.literal('JsonNull')])
  .nullable()
  .transform((v) => transformJsonNull(v));

export type NullableJsonValueType = z.infer<typeof NullableJsonValue>;

export const InputJsonValue: z.ZodType<Prisma.InputJsonValue> = z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.lazy(() => z.array(InputJsonValue.nullable())),
  z.lazy(() => z.record(InputJsonValue.nullable())),
]);

export type InputJsonValueType = z.infer<typeof InputJsonValue>;


/////////////////////////////////////////
// ENUMS
/////////////////////////////////////////

export const TransactionIsolationLevelSchema = z.enum(['ReadUncommitted','ReadCommitted','RepeatableRead','Serializable']);

export const AccountScalarFieldEnumSchema = z.enum(['id','userId','type','provider','providerAccountId','refresh_token','access_token','expires_at','token_type','scope','id_token','session_state']);

export const RelationLoadStrategySchema = z.enum(['query','join']);

export const SessionScalarFieldEnumSchema = z.enum(['id','sessionToken','userId','expires']);

export const UserSecretScalarFieldEnumSchema = z.enum(['id','username','type','value','createdAt','updatedAt']);

export const UserScalarFieldEnumSchema = z.enum(['id','name','bio','email','emailVerified','githubUsername','bot','image','admin','canTriggerExplanations','emailNewsletterNotification','emailUnsubscribeAll','emailUnsubscribeCode','createdAt']);

export const GraphMetadataSubgraphScalarFieldEnumSchema = z.enum(['id','displayName','graphMetadataId','pinnedIds','supernodes','clerps','pruningThreshold','densityThreshold','userId','isFeaturedSolution','createdAt','updatedAt']);

export const GraphMetadataScalarFieldEnumSchema = z.enum(['id','modelId','slug','promptTokens','prompt','titlePrefix','isFeatured','url','userId','createdAt','updatedAt']);

export const GraphMetadataDataPutRequestScalarFieldEnumSchema = z.enum(['id','ipAddress','filename','url','userId','createdAt']);

export const ListCommentScalarFieldEnumSchema = z.enum(['id','listId','text','userId','createdAt']);

export const ListScalarFieldEnumSchema = z.enum(['id','name','description','defaultTestText','userId','createdAt','updatedAt']);

export const ListsOnNeuronsScalarFieldEnumSchema = z.enum(['modelId','layer','index','listId','description','addedAt','userId']);

export const ListsOnActivationsScalarFieldEnumSchema = z.enum(['activationId','listId']);

export const VerificationTokenScalarFieldEnumSchema = z.enum(['identifier','token','expires']);

export const ModelScalarFieldEnumSchema = z.enum(['id','displayNameShort','displayName','creatorId','tlensId','dimension','thinking','visibility','defaultSourceSetName','defaultSourceId','inferenceEnabled','instruct','layers','neuronsPerLayer','createdAt','owner','updatedAt','website']);

export const InferenceHostSourceScalarFieldEnumSchema = z.enum(['id','name','hostUrl','modelId','createdAt','updatedAt']);

export const InferenceHostSourceOnSourceScalarFieldEnumSchema = z.enum(['sourceId','sourceModelId','inferenceHostId']);

export const SourceScalarFieldEnumSchema = z.enum(['id','modelId','hasDashboards','inferenceEnabled','saelensConfig','saelensRelease','saelensSaeId','hfRepoId','hfFolderId','visibility','defaultOfModelId','setName','creatorId','hasUmap','hasUmapLogSparsity','hasUmapClusters','num_prompts','num_tokens_in_prompt','dataset','notes','cosSimMatchModelId','cosSimMatchSourceId','createdAt']);

export const SourceSetScalarFieldEnumSchema = z.enum(['modelId','name','hasDashboards','allowInferenceSearch','visibility','description','type','creatorName','urls','creatorEmail','creatorId','releaseName','defaultOfModelId','defaultRange','defaultShowBreaks','showDfa','showCorrelated','showHeadAttribution','showUmap','createdAt']);

export const SourceReleaseScalarFieldEnumSchema = z.enum(['name','visibility','isNewUi','featured','description','descriptionShort','urls','creatorEmail','creatorName','creatorNameShort','creatorId','defaultSourceSetName','defaultSourceId','defaultUmapSourceIds','createdAt']);

export const EvalScalarFieldEnumSchema = z.enum(['id','typeName','modelId','sourceId','output','detailedMetrics','createdAt','updatedAt']);

export const EvalTypeScalarFieldEnumSchema = z.enum(['name','displayName','description','longerDescription','outputSchema','featured','url','createdAt','updatedAt']);

export const NeuronScalarFieldEnumSchema = z.enum(['modelId','layer','index','sourceSetName','creatorId','createdAt','maxActApprox','hasVector','vector','vectorLabel','vectorDefaultSteerStrength','hookName','topkCosSimIndices','topkCosSimValues','neuron_alignment_indices','neuron_alignment_values','neuron_alignment_l1','correlated_neurons_indices','correlated_neurons_pearson','correlated_neurons_l1','correlated_features_indices','correlated_features_pearson','correlated_features_l1','neg_str','neg_values','pos_str','pos_values','frac_nonzero','freq_hist_data_bar_heights','freq_hist_data_bar_values','logits_hist_data_bar_heights','logits_hist_data_bar_values','decoder_weights_dist','umap_cluster','umap_log_feature_sparsity','umap_x','umap_y']);

export const ExplanationScalarFieldEnumSchema = z.enum(['id','modelId','layer','index','description','authorId','triggeredByUserId','notes','scoreV1','scoreV2','umap_x','umap_y','umap_cluster','umap_log_feature_sparsity','typeName','explanationModelName','createdAt','updatedAt']);

export const ExplanationModelSettingsScalarFieldEnumSchema = z.enum(['id','temperature','maxTokens','topP','frequencyPenalty']);

export const ExplanationModelTypeScalarFieldEnumSchema = z.enum(['name','displayName','description','featured','openRouterModelId','url','creatorName','createdAt','updatedAt']);

export const ExplanationTypeScalarFieldEnumSchema = z.enum(['name','displayName','description','featured','isAttention','explainerModelSettingsId','settings','url','creatorName','creatorId','createdAt','updatedAt']);

export const ExplanationScoreScalarFieldEnumSchema = z.enum(['id','initiatedByUserId','value','explanationId','explanationScoreTypeName','explanationScoreModelName','jsonDetails','createdAt']);

export const ExplanationScoreTypeScalarFieldEnumSchema = z.enum(['name','displayName','description','featured','isAttention','scoreDescription','scorerModel','settings','url','creatorName','creatorId','createdAt','updatedAt']);

export const ExplanationScoreModelScalarFieldEnumSchema = z.enum(['name','displayName','description','featured','openRouterModelId','url','creatorName','createdAt','updatedAt']);

export const ExplanationActivationV1ScalarFieldEnumSchema = z.enum(['id','explanationId','activationId','expectedValues','score','scorerId','scorerAutoInterpModel','version','createdAt']);

export const VoteScalarFieldEnumSchema = z.enum(['id','voterId','reason','points','explanationId','createdAt','updatedAt']);

export const ActivationScalarFieldEnumSchema = z.enum(['id','tokens','dataIndex','index','layer','modelId','dataSource','maxValue','maxValueTokenIndex','minValue','values','dfaValues','dfaTargetIndex','dfaMaxValue','creatorId','createdAt','lossValues','logitContributions','binMin','binMax','binContains','qualifyingTokenIndex']);

export const CommentScalarFieldEnumSchema = z.enum(['id','modelId','layer','index','text','userId','createdAt']);

export const BookmarkScalarFieldEnumSchema = z.enum(['id','modelId','layer','index','userId','createdAt']);

export const SavedSearchScalarFieldEnumSchema = z.enum(['id','query','tokens','modelId','selectedLayers','sortByIndexes','ignoreBos','sourceSet','userId','updatedAt','createdAt','counts','numResults','densityThreshold']);

export const SavedSearchActivationScalarFieldEnumSchema = z.enum(['savedSearchId','activationId','order']);

export const SteerOutputToNeuronScalarFieldEnumSchema = z.enum(['modelId','layer','index','strength','steerOutputId']);

export const SteerOutputScalarFieldEnumSchema = z.enum(['id','type','modelId','steerSpecialTokens','inputText','inputTextChatTemplate','outputText','outputTextChatTemplate','temperature','numTokens','freqPenalty','seed','strengthMultiplier','steerMethod','createdAt','creatorId','version','connectedDefaultOutputId','connectedSteerOutputIds']);

export const SortOrderSchema = z.enum(['asc','desc']);

export const JsonNullValueInputSchema = z.enum(['JsonNull',]);

export const NullableJsonNullValueInputSchema = z.enum(['DbNull','JsonNull',]).transform((v) => transformJsonNull(v));

export const QueryModeSchema = z.enum(['default','insensitive']);

export const NullsOrderSchema = z.enum(['first','last']);

export const JsonNullValueFilterSchema = z.enum(['DbNull','JsonNull','AnyNull',]);

export const UserSecretTypeSchema = z.enum(['NEURONPEDIA','OPENAI','GOOGLE','ANTHROPIC','OPENROUTER']);

export type UserSecretTypeType = `${z.infer<typeof UserSecretTypeSchema>}`

export const SteerOutputTypeSchema = z.enum(['DEFAULT','STEERED']);

export type SteerOutputTypeType = `${z.infer<typeof SteerOutputTypeSchema>}`

export const VisibilitySchema = z.enum(['PUBLIC','UNLISTED','PRIVATE']);

export type VisibilityType = `${z.infer<typeof VisibilitySchema>}`

/////////////////////////////////////////
// MODELS
/////////////////////////////////////////

/////////////////////////////////////////
// ACCOUNT SCHEMA
/////////////////////////////////////////

export const AccountSchema = z.object({
  id: z.string().cuid(),
  userId: z.string(),
  type: z.string(),
  provider: z.string(),
  providerAccountId: z.string(),
  refresh_token: z.string().nullable(),
  access_token: z.string().nullable(),
  expires_at: z.number().int().nullable(),
  token_type: z.string().nullable(),
  scope: z.string().nullable(),
  id_token: z.string().nullable(),
  session_state: z.string().nullable(),
})

export type Account = z.infer<typeof AccountSchema>

/////////////////////////////////////////
// ACCOUNT PARTIAL SCHEMA
/////////////////////////////////////////

export const AccountPartialSchema = AccountSchema.partial()

export type AccountPartial = z.infer<typeof AccountPartialSchema>

// ACCOUNT RELATION SCHEMA
//------------------------------------------------------

export type AccountRelations = {
  user: UserWithRelations;
};

export type AccountWithRelations = z.infer<typeof AccountSchema> & AccountRelations

export const AccountWithRelationsSchema: z.ZodType<AccountWithRelations> = AccountSchema.merge(z.object({
  user: z.lazy(() => UserWithRelationsSchema),
}))

// ACCOUNT PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type AccountPartialRelations = {
  user?: UserPartialWithRelations;
};

export type AccountPartialWithRelations = z.infer<typeof AccountPartialSchema> & AccountPartialRelations

export const AccountPartialWithRelationsSchema: z.ZodType<AccountPartialWithRelations> = AccountPartialSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type AccountWithPartialRelations = z.infer<typeof AccountSchema> & AccountPartialRelations

export const AccountWithPartialRelationsSchema: z.ZodType<AccountWithPartialRelations> = AccountSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// SESSION SCHEMA
/////////////////////////////////////////

export const SessionSchema = z.object({
  id: z.string().cuid(),
  sessionToken: z.string(),
  userId: z.string(),
  expires: z.coerce.date(),
})

export type Session = z.infer<typeof SessionSchema>

/////////////////////////////////////////
// SESSION PARTIAL SCHEMA
/////////////////////////////////////////

export const SessionPartialSchema = SessionSchema.partial()

export type SessionPartial = z.infer<typeof SessionPartialSchema>

// SESSION RELATION SCHEMA
//------------------------------------------------------

export type SessionRelations = {
  user: UserWithRelations;
};

export type SessionWithRelations = z.infer<typeof SessionSchema> & SessionRelations

export const SessionWithRelationsSchema: z.ZodType<SessionWithRelations> = SessionSchema.merge(z.object({
  user: z.lazy(() => UserWithRelationsSchema),
}))

// SESSION PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SessionPartialRelations = {
  user?: UserPartialWithRelations;
};

export type SessionPartialWithRelations = z.infer<typeof SessionPartialSchema> & SessionPartialRelations

export const SessionPartialWithRelationsSchema: z.ZodType<SessionPartialWithRelations> = SessionPartialSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type SessionWithPartialRelations = z.infer<typeof SessionSchema> & SessionPartialRelations

export const SessionWithPartialRelationsSchema: z.ZodType<SessionWithPartialRelations> = SessionSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// USER SECRET SCHEMA
/////////////////////////////////////////

export const UserSecretSchema = z.object({
  type: UserSecretTypeSchema,
  id: z.string().cuid(),
  username: z.string(),
  value: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type UserSecret = z.infer<typeof UserSecretSchema>

/////////////////////////////////////////
// USER SECRET PARTIAL SCHEMA
/////////////////////////////////////////

export const UserSecretPartialSchema = UserSecretSchema.partial()

export type UserSecretPartial = z.infer<typeof UserSecretPartialSchema>

// USER SECRET RELATION SCHEMA
//------------------------------------------------------

export type UserSecretRelations = {
  user: UserWithRelations;
};

export type UserSecretWithRelations = z.infer<typeof UserSecretSchema> & UserSecretRelations

export const UserSecretWithRelationsSchema: z.ZodType<UserSecretWithRelations> = UserSecretSchema.merge(z.object({
  user: z.lazy(() => UserWithRelationsSchema),
}))

// USER SECRET PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type UserSecretPartialRelations = {
  user?: UserPartialWithRelations;
};

export type UserSecretPartialWithRelations = z.infer<typeof UserSecretPartialSchema> & UserSecretPartialRelations

export const UserSecretPartialWithRelationsSchema: z.ZodType<UserSecretPartialWithRelations> = UserSecretPartialSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type UserSecretWithPartialRelations = z.infer<typeof UserSecretSchema> & UserSecretPartialRelations

export const UserSecretWithPartialRelationsSchema: z.ZodType<UserSecretWithPartialRelations> = UserSecretSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// USER SCHEMA
/////////////////////////////////////////

export const UserSchema = z.object({
  id: z.string().cuid(),
  name: z.string().regex(new RegExp(/^[a-zA-Z0-9-.]+$/), "Name should contain only numbers, letters, dashes, and periods.",).min(1).max(39).transform((val) => val.toLowerCase()),
  bio: z.string().nullable(),
  email: z.string().nullable(),
  emailVerified: z.coerce.date().nullable(),
  githubUsername: z.string().nullable(),
  bot: z.boolean(),
  image: z.string().nullable(),
  admin: z.boolean(),
  canTriggerExplanations: z.boolean(),
  emailNewsletterNotification: z.boolean(),
  emailUnsubscribeAll: z.boolean(),
  emailUnsubscribeCode: z.string().cuid(),
  createdAt: z.coerce.date(),
})

export type User = z.infer<typeof UserSchema>

/////////////////////////////////////////
// USER PARTIAL SCHEMA
/////////////////////////////////////////

export const UserPartialSchema = UserSchema.partial()

export type UserPartial = z.infer<typeof UserPartialSchema>

// USER RELATION SCHEMA
//------------------------------------------------------

export type UserRelations = {
  secrets: UserSecretWithRelations[];
  accounts: AccountWithRelations[];
  explanations: ExplanationWithRelations[];
  triggeredExplanations: ExplanationWithRelations[];
  neurons: NeuronWithRelations[];
  sources: SourceWithRelations[];
  sourceSets: SourceSetWithRelations[];
  sourceReleases: SourceReleaseWithRelations[];
  models: ModelWithRelations[];
  sessions: SessionWithRelations[];
  votes: VoteWithRelations[];
  activations: ActivationWithRelations[];
  bookmarks: BookmarkWithRelations[];
  comments: CommentWithRelations[];
  lists: ListWithRelations[];
  listsOnNeurons: ListsOnNeuronsWithRelations[];
  listComments: ListCommentWithRelations[];
  savedSearches: SavedSearchWithRelations[];
  explanationActivations: ExplanationActivationV1WithRelations[];
  steerOutput: SteerOutputWithRelations[];
  explanationType: ExplanationTypeWithRelations[];
  explanationScoreType: ExplanationScoreTypeWithRelations[];
  initiatedExplanationScore: ExplanationScoreWithRelations[];
  graphMetadatas: GraphMetadataWithRelations[];
  graphMetadataDataPutRequests: GraphMetadataDataPutRequestWithRelations[];
  graphMetadataSubgraphs: GraphMetadataSubgraphWithRelations[];
};

export type UserWithRelations = z.infer<typeof UserSchema> & UserRelations

export const UserWithRelationsSchema: z.ZodType<UserWithRelations> = UserSchema.merge(z.object({
  secrets: z.lazy(() => UserSecretWithRelationsSchema).array(),
  accounts: z.lazy(() => AccountWithRelationsSchema).array(),
  explanations: z.lazy(() => ExplanationWithRelationsSchema).array(),
  triggeredExplanations: z.lazy(() => ExplanationWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronWithRelationsSchema).array(),
  sources: z.lazy(() => SourceWithRelationsSchema).array(),
  sourceSets: z.lazy(() => SourceSetWithRelationsSchema).array(),
  sourceReleases: z.lazy(() => SourceReleaseWithRelationsSchema).array(),
  models: z.lazy(() => ModelWithRelationsSchema).array(),
  sessions: z.lazy(() => SessionWithRelationsSchema).array(),
  votes: z.lazy(() => VoteWithRelationsSchema).array(),
  activations: z.lazy(() => ActivationWithRelationsSchema).array(),
  bookmarks: z.lazy(() => BookmarkWithRelationsSchema).array(),
  comments: z.lazy(() => CommentWithRelationsSchema).array(),
  lists: z.lazy(() => ListWithRelationsSchema).array(),
  listsOnNeurons: z.lazy(() => ListsOnNeuronsWithRelationsSchema).array(),
  listComments: z.lazy(() => ListCommentWithRelationsSchema).array(),
  savedSearches: z.lazy(() => SavedSearchWithRelationsSchema).array(),
  explanationActivations: z.lazy(() => ExplanationActivationV1WithRelationsSchema).array(),
  steerOutput: z.lazy(() => SteerOutputWithRelationsSchema).array(),
  explanationType: z.lazy(() => ExplanationTypeWithRelationsSchema).array(),
  explanationScoreType: z.lazy(() => ExplanationScoreTypeWithRelationsSchema).array(),
  initiatedExplanationScore: z.lazy(() => ExplanationScoreWithRelationsSchema).array(),
  graphMetadatas: z.lazy(() => GraphMetadataWithRelationsSchema).array(),
  graphMetadataDataPutRequests: z.lazy(() => GraphMetadataDataPutRequestWithRelationsSchema).array(),
  graphMetadataSubgraphs: z.lazy(() => GraphMetadataSubgraphWithRelationsSchema).array(),
}))

// USER PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type UserPartialRelations = {
  secrets?: UserSecretPartialWithRelations[];
  accounts?: AccountPartialWithRelations[];
  explanations?: ExplanationPartialWithRelations[];
  triggeredExplanations?: ExplanationPartialWithRelations[];
  neurons?: NeuronPartialWithRelations[];
  sources?: SourcePartialWithRelations[];
  sourceSets?: SourceSetPartialWithRelations[];
  sourceReleases?: SourceReleasePartialWithRelations[];
  models?: ModelPartialWithRelations[];
  sessions?: SessionPartialWithRelations[];
  votes?: VotePartialWithRelations[];
  activations?: ActivationPartialWithRelations[];
  bookmarks?: BookmarkPartialWithRelations[];
  comments?: CommentPartialWithRelations[];
  lists?: ListPartialWithRelations[];
  listsOnNeurons?: ListsOnNeuronsPartialWithRelations[];
  listComments?: ListCommentPartialWithRelations[];
  savedSearches?: SavedSearchPartialWithRelations[];
  explanationActivations?: ExplanationActivationV1PartialWithRelations[];
  steerOutput?: SteerOutputPartialWithRelations[];
  explanationType?: ExplanationTypePartialWithRelations[];
  explanationScoreType?: ExplanationScoreTypePartialWithRelations[];
  initiatedExplanationScore?: ExplanationScorePartialWithRelations[];
  graphMetadatas?: GraphMetadataPartialWithRelations[];
  graphMetadataDataPutRequests?: GraphMetadataDataPutRequestPartialWithRelations[];
  graphMetadataSubgraphs?: GraphMetadataSubgraphPartialWithRelations[];
};

export type UserPartialWithRelations = z.infer<typeof UserPartialSchema> & UserPartialRelations

export const UserPartialWithRelationsSchema: z.ZodType<UserPartialWithRelations> = UserPartialSchema.merge(z.object({
  secrets: z.lazy(() => UserSecretPartialWithRelationsSchema).array(),
  accounts: z.lazy(() => AccountPartialWithRelationsSchema).array(),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  triggeredExplanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  sources: z.lazy(() => SourcePartialWithRelationsSchema).array(),
  sourceSets: z.lazy(() => SourceSetPartialWithRelationsSchema).array(),
  sourceReleases: z.lazy(() => SourceReleasePartialWithRelationsSchema).array(),
  models: z.lazy(() => ModelPartialWithRelationsSchema).array(),
  sessions: z.lazy(() => SessionPartialWithRelationsSchema).array(),
  votes: z.lazy(() => VotePartialWithRelationsSchema).array(),
  activations: z.lazy(() => ActivationPartialWithRelationsSchema).array(),
  bookmarks: z.lazy(() => BookmarkPartialWithRelationsSchema).array(),
  comments: z.lazy(() => CommentPartialWithRelationsSchema).array(),
  lists: z.lazy(() => ListPartialWithRelationsSchema).array(),
  listsOnNeurons: z.lazy(() => ListsOnNeuronsPartialWithRelationsSchema).array(),
  listComments: z.lazy(() => ListCommentPartialWithRelationsSchema).array(),
  savedSearches: z.lazy(() => SavedSearchPartialWithRelationsSchema).array(),
  explanationActivations: z.lazy(() => ExplanationActivationV1PartialWithRelationsSchema).array(),
  steerOutput: z.lazy(() => SteerOutputPartialWithRelationsSchema).array(),
  explanationType: z.lazy(() => ExplanationTypePartialWithRelationsSchema).array(),
  explanationScoreType: z.lazy(() => ExplanationScoreTypePartialWithRelationsSchema).array(),
  initiatedExplanationScore: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
  graphMetadatas: z.lazy(() => GraphMetadataPartialWithRelationsSchema).array(),
  graphMetadataDataPutRequests: z.lazy(() => GraphMetadataDataPutRequestPartialWithRelationsSchema).array(),
  graphMetadataSubgraphs: z.lazy(() => GraphMetadataSubgraphPartialWithRelationsSchema).array(),
})).partial()

export type UserWithPartialRelations = z.infer<typeof UserSchema> & UserPartialRelations

export const UserWithPartialRelationsSchema: z.ZodType<UserWithPartialRelations> = UserSchema.merge(z.object({
  secrets: z.lazy(() => UserSecretPartialWithRelationsSchema).array(),
  accounts: z.lazy(() => AccountPartialWithRelationsSchema).array(),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  triggeredExplanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  sources: z.lazy(() => SourcePartialWithRelationsSchema).array(),
  sourceSets: z.lazy(() => SourceSetPartialWithRelationsSchema).array(),
  sourceReleases: z.lazy(() => SourceReleasePartialWithRelationsSchema).array(),
  models: z.lazy(() => ModelPartialWithRelationsSchema).array(),
  sessions: z.lazy(() => SessionPartialWithRelationsSchema).array(),
  votes: z.lazy(() => VotePartialWithRelationsSchema).array(),
  activations: z.lazy(() => ActivationPartialWithRelationsSchema).array(),
  bookmarks: z.lazy(() => BookmarkPartialWithRelationsSchema).array(),
  comments: z.lazy(() => CommentPartialWithRelationsSchema).array(),
  lists: z.lazy(() => ListPartialWithRelationsSchema).array(),
  listsOnNeurons: z.lazy(() => ListsOnNeuronsPartialWithRelationsSchema).array(),
  listComments: z.lazy(() => ListCommentPartialWithRelationsSchema).array(),
  savedSearches: z.lazy(() => SavedSearchPartialWithRelationsSchema).array(),
  explanationActivations: z.lazy(() => ExplanationActivationV1PartialWithRelationsSchema).array(),
  steerOutput: z.lazy(() => SteerOutputPartialWithRelationsSchema).array(),
  explanationType: z.lazy(() => ExplanationTypePartialWithRelationsSchema).array(),
  explanationScoreType: z.lazy(() => ExplanationScoreTypePartialWithRelationsSchema).array(),
  initiatedExplanationScore: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
  graphMetadatas: z.lazy(() => GraphMetadataPartialWithRelationsSchema).array(),
  graphMetadataDataPutRequests: z.lazy(() => GraphMetadataDataPutRequestPartialWithRelationsSchema).array(),
  graphMetadataSubgraphs: z.lazy(() => GraphMetadataSubgraphPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// GRAPH METADATA SUBGRAPH SCHEMA
/////////////////////////////////////////

export const GraphMetadataSubgraphSchema = z.object({
  id: z.string().cuid(),
  displayName: z.string().nullable(),
  graphMetadataId: z.string(),
  pinnedIds: z.string().array(),
  supernodes: InputJsonValue,
  clerps: InputJsonValue,
  pruningThreshold: z.number().nullable(),
  densityThreshold: z.number().nullable(),
  userId: z.string(),
  isFeaturedSolution: z.boolean(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type GraphMetadataSubgraph = z.infer<typeof GraphMetadataSubgraphSchema>

/////////////////////////////////////////
// GRAPH METADATA SUBGRAPH PARTIAL SCHEMA
/////////////////////////////////////////

export const GraphMetadataSubgraphPartialSchema = GraphMetadataSubgraphSchema.partial()

export type GraphMetadataSubgraphPartial = z.infer<typeof GraphMetadataSubgraphPartialSchema>

// GRAPH METADATA SUBGRAPH RELATION SCHEMA
//------------------------------------------------------

export type GraphMetadataSubgraphRelations = {
  graphMetadata: GraphMetadataWithRelations;
  user: UserWithRelations;
};

export type GraphMetadataSubgraphWithRelations = z.infer<typeof GraphMetadataSubgraphSchema> & GraphMetadataSubgraphRelations

export const GraphMetadataSubgraphWithRelationsSchema: z.ZodType<GraphMetadataSubgraphWithRelations> = GraphMetadataSubgraphSchema.merge(z.object({
  graphMetadata: z.lazy(() => GraphMetadataWithRelationsSchema),
  user: z.lazy(() => UserWithRelationsSchema),
}))

// GRAPH METADATA SUBGRAPH PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type GraphMetadataSubgraphPartialRelations = {
  graphMetadata?: GraphMetadataPartialWithRelations;
  user?: UserPartialWithRelations;
};

export type GraphMetadataSubgraphPartialWithRelations = z.infer<typeof GraphMetadataSubgraphPartialSchema> & GraphMetadataSubgraphPartialRelations

export const GraphMetadataSubgraphPartialWithRelationsSchema: z.ZodType<GraphMetadataSubgraphPartialWithRelations> = GraphMetadataSubgraphPartialSchema.merge(z.object({
  graphMetadata: z.lazy(() => GraphMetadataPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type GraphMetadataSubgraphWithPartialRelations = z.infer<typeof GraphMetadataSubgraphSchema> & GraphMetadataSubgraphPartialRelations

export const GraphMetadataSubgraphWithPartialRelationsSchema: z.ZodType<GraphMetadataSubgraphWithPartialRelations> = GraphMetadataSubgraphSchema.merge(z.object({
  graphMetadata: z.lazy(() => GraphMetadataPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// GRAPH METADATA SCHEMA
/////////////////////////////////////////

export const GraphMetadataSchema = z.object({
  id: z.string().cuid(),
  modelId: z.string(),
  slug: z.string(),
  promptTokens: z.string().array(),
  prompt: z.string(),
  titlePrefix: z.string(),
  isFeatured: z.boolean(),
  url: z.string(),
  userId: z.string().nullable(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type GraphMetadata = z.infer<typeof GraphMetadataSchema>

/////////////////////////////////////////
// GRAPH METADATA PARTIAL SCHEMA
/////////////////////////////////////////

export const GraphMetadataPartialSchema = GraphMetadataSchema.partial()

export type GraphMetadataPartial = z.infer<typeof GraphMetadataPartialSchema>

// GRAPH METADATA RELATION SCHEMA
//------------------------------------------------------

export type GraphMetadataRelations = {
  model: ModelWithRelations;
  subgraphs: GraphMetadataSubgraphWithRelations[];
  user?: UserWithRelations | null;
};

export type GraphMetadataWithRelations = z.infer<typeof GraphMetadataSchema> & GraphMetadataRelations

export const GraphMetadataWithRelationsSchema: z.ZodType<GraphMetadataWithRelations> = GraphMetadataSchema.merge(z.object({
  model: z.lazy(() => ModelWithRelationsSchema),
  subgraphs: z.lazy(() => GraphMetadataSubgraphWithRelationsSchema).array(),
  user: z.lazy(() => UserWithRelationsSchema).nullable(),
}))

// GRAPH METADATA PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type GraphMetadataPartialRelations = {
  model?: ModelPartialWithRelations;
  subgraphs?: GraphMetadataSubgraphPartialWithRelations[];
  user?: UserPartialWithRelations | null;
};

export type GraphMetadataPartialWithRelations = z.infer<typeof GraphMetadataPartialSchema> & GraphMetadataPartialRelations

export const GraphMetadataPartialWithRelationsSchema: z.ZodType<GraphMetadataPartialWithRelations> = GraphMetadataPartialSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  subgraphs: z.lazy(() => GraphMetadataSubgraphPartialWithRelationsSchema).array(),
  user: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
})).partial()

export type GraphMetadataWithPartialRelations = z.infer<typeof GraphMetadataSchema> & GraphMetadataPartialRelations

export const GraphMetadataWithPartialRelationsSchema: z.ZodType<GraphMetadataWithPartialRelations> = GraphMetadataSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  subgraphs: z.lazy(() => GraphMetadataSubgraphPartialWithRelationsSchema).array(),
  user: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
}).partial())

/////////////////////////////////////////
// GRAPH METADATA DATA PUT REQUEST SCHEMA
/////////////////////////////////////////

export const GraphMetadataDataPutRequestSchema = z.object({
  id: z.string().cuid(),
  ipAddress: z.string(),
  filename: z.string(),
  url: z.string(),
  userId: z.string(),
  createdAt: z.coerce.date(),
})

export type GraphMetadataDataPutRequest = z.infer<typeof GraphMetadataDataPutRequestSchema>

/////////////////////////////////////////
// GRAPH METADATA DATA PUT REQUEST PARTIAL SCHEMA
/////////////////////////////////////////

export const GraphMetadataDataPutRequestPartialSchema = GraphMetadataDataPutRequestSchema.partial()

export type GraphMetadataDataPutRequestPartial = z.infer<typeof GraphMetadataDataPutRequestPartialSchema>

// GRAPH METADATA DATA PUT REQUEST RELATION SCHEMA
//------------------------------------------------------

export type GraphMetadataDataPutRequestRelations = {
  user: UserWithRelations;
};

export type GraphMetadataDataPutRequestWithRelations = z.infer<typeof GraphMetadataDataPutRequestSchema> & GraphMetadataDataPutRequestRelations

export const GraphMetadataDataPutRequestWithRelationsSchema: z.ZodType<GraphMetadataDataPutRequestWithRelations> = GraphMetadataDataPutRequestSchema.merge(z.object({
  user: z.lazy(() => UserWithRelationsSchema),
}))

// GRAPH METADATA DATA PUT REQUEST PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type GraphMetadataDataPutRequestPartialRelations = {
  user?: UserPartialWithRelations;
};

export type GraphMetadataDataPutRequestPartialWithRelations = z.infer<typeof GraphMetadataDataPutRequestPartialSchema> & GraphMetadataDataPutRequestPartialRelations

export const GraphMetadataDataPutRequestPartialWithRelationsSchema: z.ZodType<GraphMetadataDataPutRequestPartialWithRelations> = GraphMetadataDataPutRequestPartialSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type GraphMetadataDataPutRequestWithPartialRelations = z.infer<typeof GraphMetadataDataPutRequestSchema> & GraphMetadataDataPutRequestPartialRelations

export const GraphMetadataDataPutRequestWithPartialRelationsSchema: z.ZodType<GraphMetadataDataPutRequestWithPartialRelations> = GraphMetadataDataPutRequestSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// LIST COMMENT SCHEMA
/////////////////////////////////////////

export const ListCommentSchema = z.object({
  id: z.string().cuid(),
  listId: z.string(),
  text: z.string(),
  userId: z.string(),
  createdAt: z.coerce.date(),
})

export type ListComment = z.infer<typeof ListCommentSchema>

/////////////////////////////////////////
// LIST COMMENT PARTIAL SCHEMA
/////////////////////////////////////////

export const ListCommentPartialSchema = ListCommentSchema.partial()

export type ListCommentPartial = z.infer<typeof ListCommentPartialSchema>

// LIST COMMENT RELATION SCHEMA
//------------------------------------------------------

export type ListCommentRelations = {
  user: UserWithRelations;
  list: ListWithRelations;
};

export type ListCommentWithRelations = z.infer<typeof ListCommentSchema> & ListCommentRelations

export const ListCommentWithRelationsSchema: z.ZodType<ListCommentWithRelations> = ListCommentSchema.merge(z.object({
  user: z.lazy(() => UserWithRelationsSchema),
  list: z.lazy(() => ListWithRelationsSchema),
}))

// LIST COMMENT PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ListCommentPartialRelations = {
  user?: UserPartialWithRelations;
  list?: ListPartialWithRelations;
};

export type ListCommentPartialWithRelations = z.infer<typeof ListCommentPartialSchema> & ListCommentPartialRelations

export const ListCommentPartialWithRelationsSchema: z.ZodType<ListCommentPartialWithRelations> = ListCommentPartialSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
  list: z.lazy(() => ListPartialWithRelationsSchema),
})).partial()

export type ListCommentWithPartialRelations = z.infer<typeof ListCommentSchema> & ListCommentPartialRelations

export const ListCommentWithPartialRelationsSchema: z.ZodType<ListCommentWithPartialRelations> = ListCommentSchema.merge(z.object({
  user: z.lazy(() => UserPartialWithRelationsSchema),
  list: z.lazy(() => ListPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// LIST SCHEMA
/////////////////////////////////////////

export const ListSchema = z.object({
  id: z.string().cuid(),
  name: z.string(),
  description: z.string(),
  defaultTestText: z.string().nullable(),
  userId: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type List = z.infer<typeof ListSchema>

/////////////////////////////////////////
// LIST PARTIAL SCHEMA
/////////////////////////////////////////

export const ListPartialSchema = ListSchema.partial()

export type ListPartial = z.infer<typeof ListPartialSchema>

// LIST RELATION SCHEMA
//------------------------------------------------------

export type ListRelations = {
  activations: ListsOnActivationsWithRelations[];
  neurons: ListsOnNeuronsWithRelations[];
  comments: ListCommentWithRelations[];
  user: UserWithRelations;
};

export type ListWithRelations = z.infer<typeof ListSchema> & ListRelations

export const ListWithRelationsSchema: z.ZodType<ListWithRelations> = ListSchema.merge(z.object({
  activations: z.lazy(() => ListsOnActivationsWithRelationsSchema).array(),
  neurons: z.lazy(() => ListsOnNeuronsWithRelationsSchema).array(),
  comments: z.lazy(() => ListCommentWithRelationsSchema).array(),
  user: z.lazy(() => UserWithRelationsSchema),
}))

// LIST PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ListPartialRelations = {
  activations?: ListsOnActivationsPartialWithRelations[];
  neurons?: ListsOnNeuronsPartialWithRelations[];
  comments?: ListCommentPartialWithRelations[];
  user?: UserPartialWithRelations;
};

export type ListPartialWithRelations = z.infer<typeof ListPartialSchema> & ListPartialRelations

export const ListPartialWithRelationsSchema: z.ZodType<ListPartialWithRelations> = ListPartialSchema.merge(z.object({
  activations: z.lazy(() => ListsOnActivationsPartialWithRelationsSchema).array(),
  neurons: z.lazy(() => ListsOnNeuronsPartialWithRelationsSchema).array(),
  comments: z.lazy(() => ListCommentPartialWithRelationsSchema).array(),
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type ListWithPartialRelations = z.infer<typeof ListSchema> & ListPartialRelations

export const ListWithPartialRelationsSchema: z.ZodType<ListWithPartialRelations> = ListSchema.merge(z.object({
  activations: z.lazy(() => ListsOnActivationsPartialWithRelationsSchema).array(),
  neurons: z.lazy(() => ListsOnNeuronsPartialWithRelationsSchema).array(),
  comments: z.lazy(() => ListCommentPartialWithRelationsSchema).array(),
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// LISTS ON NEURONS SCHEMA
/////////////////////////////////////////

export const ListsOnNeuronsSchema = z.object({
  modelId: z.string(),
  layer: z.string(),
  index: z.string(),
  listId: z.string(),
  description: z.string().nullable(),
  addedAt: z.coerce.date(),
  userId: z.string(),
})

export type ListsOnNeurons = z.infer<typeof ListsOnNeuronsSchema>

/////////////////////////////////////////
// LISTS ON NEURONS PARTIAL SCHEMA
/////////////////////////////////////////

export const ListsOnNeuronsPartialSchema = ListsOnNeuronsSchema.partial()

export type ListsOnNeuronsPartial = z.infer<typeof ListsOnNeuronsPartialSchema>

// LISTS ON NEURONS RELATION SCHEMA
//------------------------------------------------------

export type ListsOnNeuronsRelations = {
  neuron: NeuronWithRelations;
  list: ListWithRelations;
  user: UserWithRelations;
};

export type ListsOnNeuronsWithRelations = z.infer<typeof ListsOnNeuronsSchema> & ListsOnNeuronsRelations

export const ListsOnNeuronsWithRelationsSchema: z.ZodType<ListsOnNeuronsWithRelations> = ListsOnNeuronsSchema.merge(z.object({
  neuron: z.lazy(() => NeuronWithRelationsSchema),
  list: z.lazy(() => ListWithRelationsSchema),
  user: z.lazy(() => UserWithRelationsSchema),
}))

// LISTS ON NEURONS PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ListsOnNeuronsPartialRelations = {
  neuron?: NeuronPartialWithRelations;
  list?: ListPartialWithRelations;
  user?: UserPartialWithRelations;
};

export type ListsOnNeuronsPartialWithRelations = z.infer<typeof ListsOnNeuronsPartialSchema> & ListsOnNeuronsPartialRelations

export const ListsOnNeuronsPartialWithRelationsSchema: z.ZodType<ListsOnNeuronsPartialWithRelations> = ListsOnNeuronsPartialSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  list: z.lazy(() => ListPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type ListsOnNeuronsWithPartialRelations = z.infer<typeof ListsOnNeuronsSchema> & ListsOnNeuronsPartialRelations

export const ListsOnNeuronsWithPartialRelationsSchema: z.ZodType<ListsOnNeuronsWithPartialRelations> = ListsOnNeuronsSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  list: z.lazy(() => ListPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// LISTS ON ACTIVATIONS SCHEMA
/////////////////////////////////////////

export const ListsOnActivationsSchema = z.object({
  activationId: z.string(),
  listId: z.string(),
})

export type ListsOnActivations = z.infer<typeof ListsOnActivationsSchema>

/////////////////////////////////////////
// LISTS ON ACTIVATIONS PARTIAL SCHEMA
/////////////////////////////////////////

export const ListsOnActivationsPartialSchema = ListsOnActivationsSchema.partial()

export type ListsOnActivationsPartial = z.infer<typeof ListsOnActivationsPartialSchema>

// LISTS ON ACTIVATIONS RELATION SCHEMA
//------------------------------------------------------

export type ListsOnActivationsRelations = {
  activation: ActivationWithRelations;
  list: ListWithRelations;
};

export type ListsOnActivationsWithRelations = z.infer<typeof ListsOnActivationsSchema> & ListsOnActivationsRelations

export const ListsOnActivationsWithRelationsSchema: z.ZodType<ListsOnActivationsWithRelations> = ListsOnActivationsSchema.merge(z.object({
  activation: z.lazy(() => ActivationWithRelationsSchema),
  list: z.lazy(() => ListWithRelationsSchema),
}))

// LISTS ON ACTIVATIONS PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ListsOnActivationsPartialRelations = {
  activation?: ActivationPartialWithRelations;
  list?: ListPartialWithRelations;
};

export type ListsOnActivationsPartialWithRelations = z.infer<typeof ListsOnActivationsPartialSchema> & ListsOnActivationsPartialRelations

export const ListsOnActivationsPartialWithRelationsSchema: z.ZodType<ListsOnActivationsPartialWithRelations> = ListsOnActivationsPartialSchema.merge(z.object({
  activation: z.lazy(() => ActivationPartialWithRelationsSchema),
  list: z.lazy(() => ListPartialWithRelationsSchema),
})).partial()

export type ListsOnActivationsWithPartialRelations = z.infer<typeof ListsOnActivationsSchema> & ListsOnActivationsPartialRelations

export const ListsOnActivationsWithPartialRelationsSchema: z.ZodType<ListsOnActivationsWithPartialRelations> = ListsOnActivationsSchema.merge(z.object({
  activation: z.lazy(() => ActivationPartialWithRelationsSchema),
  list: z.lazy(() => ListPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// VERIFICATION TOKEN SCHEMA
/////////////////////////////////////////

export const VerificationTokenSchema = z.object({
  identifier: z.string(),
  token: z.string(),
  expires: z.coerce.date(),
})

export type VerificationToken = z.infer<typeof VerificationTokenSchema>

/////////////////////////////////////////
// VERIFICATION TOKEN PARTIAL SCHEMA
/////////////////////////////////////////

export const VerificationTokenPartialSchema = VerificationTokenSchema.partial()

export type VerificationTokenPartial = z.infer<typeof VerificationTokenPartialSchema>

/////////////////////////////////////////
// MODEL SCHEMA
/////////////////////////////////////////

export const ModelSchema = z.object({
  visibility: VisibilitySchema,
  id: z.string().regex(new RegExp(/^[a-zA-Z0-9-_]+$/), "Name should contain only numbers, letters, underscores, or dashes",).min(1).max(128).transform((val) => val.toLowerCase()),
  displayNameShort: z.string(),
  displayName: z.string(),
  creatorId: z.string(),
  tlensId: z.string().nullable(),
  dimension: z.number().int().nullable(),
  thinking: z.boolean(),
  defaultSourceSetName: z.string().nullable(),
  defaultSourceId: z.string().nullable(),
  inferenceEnabled: z.boolean(),
  instruct: z.boolean(),
  layers: z.number().int().min(0),
  neuronsPerLayer: z.number().int().min(0),
  createdAt: z.coerce.date(),
  owner: z.string().trim().min(1),
  updatedAt: z.coerce.date(),
  website: z.string().url().nullable(),
})

export type Model = z.infer<typeof ModelSchema>

/////////////////////////////////////////
// MODEL PARTIAL SCHEMA
/////////////////////////////////////////

export const ModelPartialSchema = ModelSchema.partial()

export type ModelPartial = z.infer<typeof ModelPartialSchema>

// MODEL RELATION SCHEMA
//------------------------------------------------------

export type ModelRelations = {
  defaultSourceSet?: SourceSetWithRelations | null;
  defaultSource?: SourceWithRelations | null;
  explanations: ExplanationWithRelations[];
  creator: UserWithRelations;
  neurons: NeuronWithRelations[];
  savedSearches: SavedSearchWithRelations[];
  sourceSets: SourceSetWithRelations[];
  steerOutputs: SteerOutputWithRelations[];
  evals: EvalWithRelations[];
  sourceInferenceHosts: InferenceHostSourceWithRelations[];
  graphMetadata: GraphMetadataWithRelations[];
};

export type ModelWithRelations = z.infer<typeof ModelSchema> & ModelRelations

export const ModelWithRelationsSchema: z.ZodType<ModelWithRelations> = ModelSchema.merge(z.object({
  defaultSourceSet: z.lazy(() => SourceSetWithRelationsSchema).nullable(),
  defaultSource: z.lazy(() => SourceWithRelationsSchema).nullable(),
  explanations: z.lazy(() => ExplanationWithRelationsSchema).array(),
  creator: z.lazy(() => UserWithRelationsSchema),
  neurons: z.lazy(() => NeuronWithRelationsSchema).array(),
  savedSearches: z.lazy(() => SavedSearchWithRelationsSchema).array(),
  sourceSets: z.lazy(() => SourceSetWithRelationsSchema).array(),
  steerOutputs: z.lazy(() => SteerOutputWithRelationsSchema).array(),
  evals: z.lazy(() => EvalWithRelationsSchema).array(),
  sourceInferenceHosts: z.lazy(() => InferenceHostSourceWithRelationsSchema).array(),
  graphMetadata: z.lazy(() => GraphMetadataWithRelationsSchema).array(),
}))

// MODEL PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ModelPartialRelations = {
  defaultSourceSet?: SourceSetPartialWithRelations | null;
  defaultSource?: SourcePartialWithRelations | null;
  explanations?: ExplanationPartialWithRelations[];
  creator?: UserPartialWithRelations;
  neurons?: NeuronPartialWithRelations[];
  savedSearches?: SavedSearchPartialWithRelations[];
  sourceSets?: SourceSetPartialWithRelations[];
  steerOutputs?: SteerOutputPartialWithRelations[];
  evals?: EvalPartialWithRelations[];
  sourceInferenceHosts?: InferenceHostSourcePartialWithRelations[];
  graphMetadata?: GraphMetadataPartialWithRelations[];
};

export type ModelPartialWithRelations = z.infer<typeof ModelPartialSchema> & ModelPartialRelations

export const ModelPartialWithRelationsSchema: z.ZodType<ModelPartialWithRelations> = ModelPartialSchema.merge(z.object({
  defaultSourceSet: z.lazy(() => SourceSetPartialWithRelationsSchema).nullable(),
  defaultSource: z.lazy(() => SourcePartialWithRelationsSchema).nullable(),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  savedSearches: z.lazy(() => SavedSearchPartialWithRelationsSchema).array(),
  sourceSets: z.lazy(() => SourceSetPartialWithRelationsSchema).array(),
  steerOutputs: z.lazy(() => SteerOutputPartialWithRelationsSchema).array(),
  evals: z.lazy(() => EvalPartialWithRelationsSchema).array(),
  sourceInferenceHosts: z.lazy(() => InferenceHostSourcePartialWithRelationsSchema).array(),
  graphMetadata: z.lazy(() => GraphMetadataPartialWithRelationsSchema).array(),
})).partial()

export type ModelWithPartialRelations = z.infer<typeof ModelSchema> & ModelPartialRelations

export const ModelWithPartialRelationsSchema: z.ZodType<ModelWithPartialRelations> = ModelSchema.merge(z.object({
  defaultSourceSet: z.lazy(() => SourceSetPartialWithRelationsSchema).nullable(),
  defaultSource: z.lazy(() => SourcePartialWithRelationsSchema).nullable(),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  savedSearches: z.lazy(() => SavedSearchPartialWithRelationsSchema).array(),
  sourceSets: z.lazy(() => SourceSetPartialWithRelationsSchema).array(),
  steerOutputs: z.lazy(() => SteerOutputPartialWithRelationsSchema).array(),
  evals: z.lazy(() => EvalPartialWithRelationsSchema).array(),
  sourceInferenceHosts: z.lazy(() => InferenceHostSourcePartialWithRelationsSchema).array(),
  graphMetadata: z.lazy(() => GraphMetadataPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// INFERENCE HOST SOURCE SCHEMA
/////////////////////////////////////////

export const InferenceHostSourceSchema = z.object({
  id: z.string().cuid(),
  name: z.string(),
  hostUrl: z.string(),
  modelId: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type InferenceHostSource = z.infer<typeof InferenceHostSourceSchema>

/////////////////////////////////////////
// INFERENCE HOST SOURCE PARTIAL SCHEMA
/////////////////////////////////////////

export const InferenceHostSourcePartialSchema = InferenceHostSourceSchema.partial()

export type InferenceHostSourcePartial = z.infer<typeof InferenceHostSourcePartialSchema>

// INFERENCE HOST SOURCE RELATION SCHEMA
//------------------------------------------------------

export type InferenceHostSourceRelations = {
  model: ModelWithRelations;
  sources: InferenceHostSourceOnSourceWithRelations[];
};

export type InferenceHostSourceWithRelations = z.infer<typeof InferenceHostSourceSchema> & InferenceHostSourceRelations

export const InferenceHostSourceWithRelationsSchema: z.ZodType<InferenceHostSourceWithRelations> = InferenceHostSourceSchema.merge(z.object({
  model: z.lazy(() => ModelWithRelationsSchema),
  sources: z.lazy(() => InferenceHostSourceOnSourceWithRelationsSchema).array(),
}))

// INFERENCE HOST SOURCE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type InferenceHostSourcePartialRelations = {
  model?: ModelPartialWithRelations;
  sources?: InferenceHostSourceOnSourcePartialWithRelations[];
};

export type InferenceHostSourcePartialWithRelations = z.infer<typeof InferenceHostSourcePartialSchema> & InferenceHostSourcePartialRelations

export const InferenceHostSourcePartialWithRelationsSchema: z.ZodType<InferenceHostSourcePartialWithRelations> = InferenceHostSourcePartialSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  sources: z.lazy(() => InferenceHostSourceOnSourcePartialWithRelationsSchema).array(),
})).partial()

export type InferenceHostSourceWithPartialRelations = z.infer<typeof InferenceHostSourceSchema> & InferenceHostSourcePartialRelations

export const InferenceHostSourceWithPartialRelationsSchema: z.ZodType<InferenceHostSourceWithPartialRelations> = InferenceHostSourceSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  sources: z.lazy(() => InferenceHostSourceOnSourcePartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// INFERENCE HOST SOURCE ON SOURCE SCHEMA
/////////////////////////////////////////

export const InferenceHostSourceOnSourceSchema = z.object({
  sourceId: z.string(),
  sourceModelId: z.string(),
  inferenceHostId: z.string(),
})

export type InferenceHostSourceOnSource = z.infer<typeof InferenceHostSourceOnSourceSchema>

/////////////////////////////////////////
// INFERENCE HOST SOURCE ON SOURCE PARTIAL SCHEMA
/////////////////////////////////////////

export const InferenceHostSourceOnSourcePartialSchema = InferenceHostSourceOnSourceSchema.partial()

export type InferenceHostSourceOnSourcePartial = z.infer<typeof InferenceHostSourceOnSourcePartialSchema>

// INFERENCE HOST SOURCE ON SOURCE RELATION SCHEMA
//------------------------------------------------------

export type InferenceHostSourceOnSourceRelations = {
  source: SourceWithRelations;
  inferenceHost: InferenceHostSourceWithRelations;
};

export type InferenceHostSourceOnSourceWithRelations = z.infer<typeof InferenceHostSourceOnSourceSchema> & InferenceHostSourceOnSourceRelations

export const InferenceHostSourceOnSourceWithRelationsSchema: z.ZodType<InferenceHostSourceOnSourceWithRelations> = InferenceHostSourceOnSourceSchema.merge(z.object({
  source: z.lazy(() => SourceWithRelationsSchema),
  inferenceHost: z.lazy(() => InferenceHostSourceWithRelationsSchema),
}))

// INFERENCE HOST SOURCE ON SOURCE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type InferenceHostSourceOnSourcePartialRelations = {
  source?: SourcePartialWithRelations;
  inferenceHost?: InferenceHostSourcePartialWithRelations;
};

export type InferenceHostSourceOnSourcePartialWithRelations = z.infer<typeof InferenceHostSourceOnSourcePartialSchema> & InferenceHostSourceOnSourcePartialRelations

export const InferenceHostSourceOnSourcePartialWithRelationsSchema: z.ZodType<InferenceHostSourceOnSourcePartialWithRelations> = InferenceHostSourceOnSourcePartialSchema.merge(z.object({
  source: z.lazy(() => SourcePartialWithRelationsSchema),
  inferenceHost: z.lazy(() => InferenceHostSourcePartialWithRelationsSchema),
})).partial()

export type InferenceHostSourceOnSourceWithPartialRelations = z.infer<typeof InferenceHostSourceOnSourceSchema> & InferenceHostSourceOnSourcePartialRelations

export const InferenceHostSourceOnSourceWithPartialRelationsSchema: z.ZodType<InferenceHostSourceOnSourceWithPartialRelations> = InferenceHostSourceOnSourceSchema.merge(z.object({
  source: z.lazy(() => SourcePartialWithRelationsSchema),
  inferenceHost: z.lazy(() => InferenceHostSourcePartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// SOURCE SCHEMA
/////////////////////////////////////////

export const SourceSchema = z.object({
  visibility: VisibilitySchema,
  id: z.string(),
  modelId: z.string(),
  hasDashboards: z.boolean(),
  inferenceEnabled: z.boolean(),
  saelensConfig: NullableJsonValue.optional(),
  saelensRelease: z.string().nullable(),
  saelensSaeId: z.string().nullable(),
  hfRepoId: z.string().nullable(),
  hfFolderId: z.string().nullable(),
  defaultOfModelId: z.string().nullable(),
  setName: z.string(),
  creatorId: z.string(),
  hasUmap: z.boolean(),
  hasUmapLogSparsity: z.boolean(),
  hasUmapClusters: z.boolean(),
  num_prompts: z.number().int().nullable(),
  num_tokens_in_prompt: z.number().int().nullable(),
  dataset: z.string().nullable(),
  notes: z.string().nullable(),
  cosSimMatchModelId: z.string().nullable(),
  cosSimMatchSourceId: z.string().nullable(),
  createdAt: z.coerce.date(),
})

export type Source = z.infer<typeof SourceSchema>

/////////////////////////////////////////
// SOURCE PARTIAL SCHEMA
/////////////////////////////////////////

export const SourcePartialSchema = SourceSchema.partial()

export type SourcePartial = z.infer<typeof SourcePartialSchema>

// SOURCE RELATION SCHEMA
//------------------------------------------------------

export type SourceRelations = {
  inferenceHosts: InferenceHostSourceOnSourceWithRelations[];
  neurons: NeuronWithRelations[];
  defaultOfModel?: ModelWithRelations | null;
  set: SourceSetWithRelations;
  creator: UserWithRelations;
  cosSimMatchSource?: SourceWithRelations | null;
  matchedBySources: SourceWithRelations[];
  evals: EvalWithRelations[];
};

export type SourceWithRelations = Omit<z.infer<typeof SourceSchema>, "saelensConfig"> & {
  saelensConfig?: NullableJsonInput;
} & SourceRelations

export const SourceWithRelationsSchema: z.ZodType<SourceWithRelations> = SourceSchema.merge(z.object({
  inferenceHosts: z.lazy(() => InferenceHostSourceOnSourceWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronWithRelationsSchema).array(),
  defaultOfModel: z.lazy(() => ModelWithRelationsSchema).nullable(),
  set: z.lazy(() => SourceSetWithRelationsSchema),
  creator: z.lazy(() => UserWithRelationsSchema),
  cosSimMatchSource: z.lazy(() => SourceWithRelationsSchema).nullable(),
  matchedBySources: z.lazy(() => SourceWithRelationsSchema).array(),
  evals: z.lazy(() => EvalWithRelationsSchema).array(),
}))

// SOURCE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SourcePartialRelations = {
  inferenceHosts?: InferenceHostSourceOnSourcePartialWithRelations[];
  neurons?: NeuronPartialWithRelations[];
  defaultOfModel?: ModelPartialWithRelations | null;
  set?: SourceSetPartialWithRelations;
  creator?: UserPartialWithRelations;
  cosSimMatchSource?: SourcePartialWithRelations | null;
  matchedBySources?: SourcePartialWithRelations[];
  evals?: EvalPartialWithRelations[];
};

export type SourcePartialWithRelations = Omit<z.infer<typeof SourcePartialSchema>, "saelensConfig"> & {
  saelensConfig?: NullableJsonInput;
} & SourcePartialRelations

export const SourcePartialWithRelationsSchema: z.ZodType<SourcePartialWithRelations> = SourcePartialSchema.merge(z.object({
  inferenceHosts: z.lazy(() => InferenceHostSourceOnSourcePartialWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  defaultOfModel: z.lazy(() => ModelPartialWithRelationsSchema).nullable(),
  set: z.lazy(() => SourceSetPartialWithRelationsSchema),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  cosSimMatchSource: z.lazy(() => SourcePartialWithRelationsSchema).nullable(),
  matchedBySources: z.lazy(() => SourcePartialWithRelationsSchema).array(),
  evals: z.lazy(() => EvalPartialWithRelationsSchema).array(),
})).partial()

export type SourceWithPartialRelations = Omit<z.infer<typeof SourceSchema>, "saelensConfig"> & {
  saelensConfig?: NullableJsonInput;
} & SourcePartialRelations

export const SourceWithPartialRelationsSchema: z.ZodType<SourceWithPartialRelations> = SourceSchema.merge(z.object({
  inferenceHosts: z.lazy(() => InferenceHostSourceOnSourcePartialWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  defaultOfModel: z.lazy(() => ModelPartialWithRelationsSchema).nullable(),
  set: z.lazy(() => SourceSetPartialWithRelationsSchema),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  cosSimMatchSource: z.lazy(() => SourcePartialWithRelationsSchema).nullable(),
  matchedBySources: z.lazy(() => SourcePartialWithRelationsSchema).array(),
  evals: z.lazy(() => EvalPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// SOURCE SET SCHEMA
/////////////////////////////////////////

export const SourceSetSchema = z.object({
  visibility: VisibilitySchema,
  modelId: z.string(),
  name: z.string(),
  hasDashboards: z.boolean(),
  allowInferenceSearch: z.boolean(),
  description: z.string(),
  type: z.string(),
  creatorName: z.string(),
  urls: z.string().array(),
  creatorEmail: z.string().nullable(),
  creatorId: z.string(),
  releaseName: z.string().nullable(),
  defaultOfModelId: z.string().nullable(),
  defaultRange: z.number().int(),
  defaultShowBreaks: z.boolean(),
  showDfa: z.boolean(),
  showCorrelated: z.boolean(),
  showHeadAttribution: z.boolean(),
  showUmap: z.boolean(),
  createdAt: z.coerce.date(),
})

export type SourceSet = z.infer<typeof SourceSetSchema>

/////////////////////////////////////////
// SOURCE SET PARTIAL SCHEMA
/////////////////////////////////////////

export const SourceSetPartialSchema = SourceSetSchema.partial()

export type SourceSetPartial = z.infer<typeof SourceSetPartialSchema>

// SOURCE SET RELATION SCHEMA
//------------------------------------------------------

export type SourceSetRelations = {
  model: ModelWithRelations;
  creator: UserWithRelations;
  sources: SourceWithRelations[];
  neurons: NeuronWithRelations[];
  releases?: SourceReleaseWithRelations | null;
  defaultOfModel?: ModelWithRelations | null;
};

export type SourceSetWithRelations = z.infer<typeof SourceSetSchema> & SourceSetRelations

export const SourceSetWithRelationsSchema: z.ZodType<SourceSetWithRelations> = SourceSetSchema.merge(z.object({
  model: z.lazy(() => ModelWithRelationsSchema),
  creator: z.lazy(() => UserWithRelationsSchema),
  sources: z.lazy(() => SourceWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronWithRelationsSchema).array(),
  releases: z.lazy(() => SourceReleaseWithRelationsSchema).nullable(),
  defaultOfModel: z.lazy(() => ModelWithRelationsSchema).nullable(),
}))

// SOURCE SET PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SourceSetPartialRelations = {
  model?: ModelPartialWithRelations;
  creator?: UserPartialWithRelations;
  sources?: SourcePartialWithRelations[];
  neurons?: NeuronPartialWithRelations[];
  releases?: SourceReleasePartialWithRelations | null;
  defaultOfModel?: ModelPartialWithRelations | null;
};

export type SourceSetPartialWithRelations = z.infer<typeof SourceSetPartialSchema> & SourceSetPartialRelations

export const SourceSetPartialWithRelationsSchema: z.ZodType<SourceSetPartialWithRelations> = SourceSetPartialSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  sources: z.lazy(() => SourcePartialWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  releases: z.lazy(() => SourceReleasePartialWithRelationsSchema).nullable(),
  defaultOfModel: z.lazy(() => ModelPartialWithRelationsSchema).nullable(),
})).partial()

export type SourceSetWithPartialRelations = z.infer<typeof SourceSetSchema> & SourceSetPartialRelations

export const SourceSetWithPartialRelationsSchema: z.ZodType<SourceSetWithPartialRelations> = SourceSetSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  sources: z.lazy(() => SourcePartialWithRelationsSchema).array(),
  neurons: z.lazy(() => NeuronPartialWithRelationsSchema).array(),
  releases: z.lazy(() => SourceReleasePartialWithRelationsSchema).nullable(),
  defaultOfModel: z.lazy(() => ModelPartialWithRelationsSchema).nullable(),
}).partial())

/////////////////////////////////////////
// SOURCE RELEASE SCHEMA
/////////////////////////////////////////

export const SourceReleaseSchema = z.object({
  visibility: VisibilitySchema,
  name: z.string(),
  isNewUi: z.boolean(),
  featured: z.boolean(),
  description: z.string(),
  descriptionShort: z.string().nullable(),
  urls: z.string().array(),
  creatorEmail: z.string().nullable(),
  creatorName: z.string(),
  creatorNameShort: z.string().nullable(),
  creatorId: z.string(),
  defaultSourceSetName: z.string().nullable(),
  defaultSourceId: z.string().nullable(),
  defaultUmapSourceIds: z.string().array(),
  createdAt: z.coerce.date(),
})

export type SourceRelease = z.infer<typeof SourceReleaseSchema>

/////////////////////////////////////////
// SOURCE RELEASE PARTIAL SCHEMA
/////////////////////////////////////////

export const SourceReleasePartialSchema = SourceReleaseSchema.partial()

export type SourceReleasePartial = z.infer<typeof SourceReleasePartialSchema>

// SOURCE RELEASE RELATION SCHEMA
//------------------------------------------------------

export type SourceReleaseRelations = {
  sourceSets: SourceSetWithRelations[];
  creator: UserWithRelations;
};

export type SourceReleaseWithRelations = z.infer<typeof SourceReleaseSchema> & SourceReleaseRelations

export const SourceReleaseWithRelationsSchema: z.ZodType<SourceReleaseWithRelations> = SourceReleaseSchema.merge(z.object({
  sourceSets: z.lazy(() => SourceSetWithRelationsSchema).array(),
  creator: z.lazy(() => UserWithRelationsSchema),
}))

// SOURCE RELEASE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SourceReleasePartialRelations = {
  sourceSets?: SourceSetPartialWithRelations[];
  creator?: UserPartialWithRelations;
};

export type SourceReleasePartialWithRelations = z.infer<typeof SourceReleasePartialSchema> & SourceReleasePartialRelations

export const SourceReleasePartialWithRelationsSchema: z.ZodType<SourceReleasePartialWithRelations> = SourceReleasePartialSchema.merge(z.object({
  sourceSets: z.lazy(() => SourceSetPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type SourceReleaseWithPartialRelations = z.infer<typeof SourceReleaseSchema> & SourceReleasePartialRelations

export const SourceReleaseWithPartialRelationsSchema: z.ZodType<SourceReleaseWithPartialRelations> = SourceReleaseSchema.merge(z.object({
  sourceSets: z.lazy(() => SourceSetPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// EVAL SCHEMA
/////////////////////////////////////////

export const EvalSchema = z.object({
  id: z.string().cuid(),
  typeName: z.string(),
  modelId: z.string(),
  sourceId: z.string(),
  output: InputJsonValue,
  detailedMetrics: NullableJsonValue.optional(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type Eval = z.infer<typeof EvalSchema>

/////////////////////////////////////////
// EVAL PARTIAL SCHEMA
/////////////////////////////////////////

export const EvalPartialSchema = EvalSchema.partial()

export type EvalPartial = z.infer<typeof EvalPartialSchema>

// EVAL RELATION SCHEMA
//------------------------------------------------------

export type EvalRelations = {
  type: EvalTypeWithRelations;
  model: ModelWithRelations;
  source: SourceWithRelations;
};

export type EvalWithRelations = Omit<z.infer<typeof EvalSchema>, "detailedMetrics"> & {
  detailedMetrics?: NullableJsonInput;
} & EvalRelations

export const EvalWithRelationsSchema: z.ZodType<EvalWithRelations> = EvalSchema.merge(z.object({
  type: z.lazy(() => EvalTypeWithRelationsSchema),
  model: z.lazy(() => ModelWithRelationsSchema),
  source: z.lazy(() => SourceWithRelationsSchema),
}))

// EVAL PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type EvalPartialRelations = {
  type?: EvalTypePartialWithRelations;
  model?: ModelPartialWithRelations;
  source?: SourcePartialWithRelations;
};

export type EvalPartialWithRelations = Omit<z.infer<typeof EvalPartialSchema>, "detailedMetrics"> & {
  detailedMetrics?: NullableJsonInput;
} & EvalPartialRelations

export const EvalPartialWithRelationsSchema: z.ZodType<EvalPartialWithRelations> = EvalPartialSchema.merge(z.object({
  type: z.lazy(() => EvalTypePartialWithRelationsSchema),
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  source: z.lazy(() => SourcePartialWithRelationsSchema),
})).partial()

export type EvalWithPartialRelations = Omit<z.infer<typeof EvalSchema>, "detailedMetrics"> & {
  detailedMetrics?: NullableJsonInput;
} & EvalPartialRelations

export const EvalWithPartialRelationsSchema: z.ZodType<EvalWithPartialRelations> = EvalSchema.merge(z.object({
  type: z.lazy(() => EvalTypePartialWithRelationsSchema),
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  source: z.lazy(() => SourcePartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// EVAL TYPE SCHEMA
/////////////////////////////////////////

export const EvalTypeSchema = z.object({
  name: z.string(),
  displayName: z.string(),
  description: z.string(),
  longerDescription: z.string(),
  outputSchema: InputJsonValue,
  featured: z.boolean(),
  url: z.string().nullable(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type EvalType = z.infer<typeof EvalTypeSchema>

/////////////////////////////////////////
// EVAL TYPE PARTIAL SCHEMA
/////////////////////////////////////////

export const EvalTypePartialSchema = EvalTypeSchema.partial()

export type EvalTypePartial = z.infer<typeof EvalTypePartialSchema>

// EVAL TYPE RELATION SCHEMA
//------------------------------------------------------

export type EvalTypeRelations = {
  evals: EvalWithRelations[];
};

export type EvalTypeWithRelations = z.infer<typeof EvalTypeSchema> & EvalTypeRelations

export const EvalTypeWithRelationsSchema: z.ZodType<EvalTypeWithRelations> = EvalTypeSchema.merge(z.object({
  evals: z.lazy(() => EvalWithRelationsSchema).array(),
}))

// EVAL TYPE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type EvalTypePartialRelations = {
  evals?: EvalPartialWithRelations[];
};

export type EvalTypePartialWithRelations = z.infer<typeof EvalTypePartialSchema> & EvalTypePartialRelations

export const EvalTypePartialWithRelationsSchema: z.ZodType<EvalTypePartialWithRelations> = EvalTypePartialSchema.merge(z.object({
  evals: z.lazy(() => EvalPartialWithRelationsSchema).array(),
})).partial()

export type EvalTypeWithPartialRelations = z.infer<typeof EvalTypeSchema> & EvalTypePartialRelations

export const EvalTypeWithPartialRelationsSchema: z.ZodType<EvalTypeWithPartialRelations> = EvalTypeSchema.merge(z.object({
  evals: z.lazy(() => EvalPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// NEURON SCHEMA
/////////////////////////////////////////

export const NeuronSchema = z.object({
  modelId: z.string(),
  layer: z.string(),
  index: z.string(),
  sourceSetName: z.string().nullable(),
  creatorId: z.string().nullable(),
  createdAt: z.coerce.date(),
  maxActApprox: z.number().nullable(),
  hasVector: z.boolean(),
  vector: z.number().array(),
  vectorLabel: z.string().nullable(),
  vectorDefaultSteerStrength: z.number().nullable(),
  hookName: z.string().nullable(),
  topkCosSimIndices: z.number().int().array(),
  topkCosSimValues: z.number().array(),
  neuron_alignment_indices: z.number().int().array(),
  neuron_alignment_values: z.number().array(),
  neuron_alignment_l1: z.number().array(),
  correlated_neurons_indices: z.number().int().array(),
  correlated_neurons_pearson: z.number().array(),
  correlated_neurons_l1: z.number().array(),
  correlated_features_indices: z.number().int().array(),
  correlated_features_pearson: z.number().array(),
  correlated_features_l1: z.number().array(),
  neg_str: z.string().array(),
  neg_values: z.number().array(),
  pos_str: z.string().array(),
  pos_values: z.number().array(),
  frac_nonzero: z.number(),
  freq_hist_data_bar_heights: z.number().array(),
  freq_hist_data_bar_values: z.number().array(),
  logits_hist_data_bar_heights: z.number().array(),
  logits_hist_data_bar_values: z.number().array(),
  decoder_weights_dist: z.number().array(),
  umap_cluster: z.number().int().nullable(),
  umap_log_feature_sparsity: z.number().nullable(),
  umap_x: z.number().nullable(),
  umap_y: z.number().nullable(),
})

export type Neuron = z.infer<typeof NeuronSchema>

/////////////////////////////////////////
// NEURON PARTIAL SCHEMA
/////////////////////////////////////////

export const NeuronPartialSchema = NeuronSchema.partial()

export type NeuronPartial = z.infer<typeof NeuronPartialSchema>

// NEURON RELATION SCHEMA
//------------------------------------------------------

export type NeuronRelations = {
  lists: ListsOnNeuronsWithRelations[];
  source?: SourceWithRelations | null;
  sourceSet?: SourceSetWithRelations | null;
  creator?: UserWithRelations | null;
  activations: ActivationWithRelations[];
  explanations: ExplanationWithRelations[];
  comments: CommentWithRelations[];
  bookmarks: BookmarkWithRelations[];
  model: ModelWithRelations;
  steerOutputs: SteerOutputToNeuronWithRelations[];
};

export type NeuronWithRelations = z.infer<typeof NeuronSchema> & NeuronRelations

export const NeuronWithRelationsSchema: z.ZodType<NeuronWithRelations> = NeuronSchema.merge(z.object({
  lists: z.lazy(() => ListsOnNeuronsWithRelationsSchema).array(),
  source: z.lazy(() => SourceWithRelationsSchema).nullable(),
  sourceSet: z.lazy(() => SourceSetWithRelationsSchema).nullable(),
  creator: z.lazy(() => UserWithRelationsSchema).nullable(),
  activations: z.lazy(() => ActivationWithRelationsSchema).array(),
  explanations: z.lazy(() => ExplanationWithRelationsSchema).array(),
  comments: z.lazy(() => CommentWithRelationsSchema).array(),
  bookmarks: z.lazy(() => BookmarkWithRelationsSchema).array(),
  model: z.lazy(() => ModelWithRelationsSchema),
  steerOutputs: z.lazy(() => SteerOutputToNeuronWithRelationsSchema).array(),
}))

// NEURON PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type NeuronPartialRelations = {
  lists?: ListsOnNeuronsPartialWithRelations[];
  source?: SourcePartialWithRelations | null;
  sourceSet?: SourceSetPartialWithRelations | null;
  creator?: UserPartialWithRelations | null;
  activations?: ActivationPartialWithRelations[];
  explanations?: ExplanationPartialWithRelations[];
  comments?: CommentPartialWithRelations[];
  bookmarks?: BookmarkPartialWithRelations[];
  model?: ModelPartialWithRelations;
  steerOutputs?: SteerOutputToNeuronPartialWithRelations[];
};

export type NeuronPartialWithRelations = z.infer<typeof NeuronPartialSchema> & NeuronPartialRelations

export const NeuronPartialWithRelationsSchema: z.ZodType<NeuronPartialWithRelations> = NeuronPartialSchema.merge(z.object({
  lists: z.lazy(() => ListsOnNeuronsPartialWithRelationsSchema).array(),
  source: z.lazy(() => SourcePartialWithRelationsSchema).nullable(),
  sourceSet: z.lazy(() => SourceSetPartialWithRelationsSchema).nullable(),
  creator: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
  activations: z.lazy(() => ActivationPartialWithRelationsSchema).array(),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  comments: z.lazy(() => CommentPartialWithRelationsSchema).array(),
  bookmarks: z.lazy(() => BookmarkPartialWithRelationsSchema).array(),
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  steerOutputs: z.lazy(() => SteerOutputToNeuronPartialWithRelationsSchema).array(),
})).partial()

export type NeuronWithPartialRelations = z.infer<typeof NeuronSchema> & NeuronPartialRelations

export const NeuronWithPartialRelationsSchema: z.ZodType<NeuronWithPartialRelations> = NeuronSchema.merge(z.object({
  lists: z.lazy(() => ListsOnNeuronsPartialWithRelationsSchema).array(),
  source: z.lazy(() => SourcePartialWithRelationsSchema).nullable(),
  sourceSet: z.lazy(() => SourceSetPartialWithRelationsSchema).nullable(),
  creator: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
  activations: z.lazy(() => ActivationPartialWithRelationsSchema).array(),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
  comments: z.lazy(() => CommentPartialWithRelationsSchema).array(),
  bookmarks: z.lazy(() => BookmarkPartialWithRelationsSchema).array(),
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  steerOutputs: z.lazy(() => SteerOutputToNeuronPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// EXPLANATION SCHEMA
/////////////////////////////////////////

export const ExplanationSchema = z.object({
  id: z.string().cuid(),
  modelId: z.string(),
  layer: z.string(),
  index: z.string(),
  description: z.string(),
  authorId: z.string(),
  triggeredByUserId: z.string().nullable(),
  notes: z.string().nullable(),
  scoreV1: z.number(),
  scoreV2: z.number().nullable(),
  umap_x: z.number().nullable(),
  umap_y: z.number().nullable(),
  umap_cluster: z.number().int().nullable(),
  umap_log_feature_sparsity: z.number().nullable(),
  typeName: z.string().nullable(),
  explanationModelName: z.string().nullable(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type Explanation = z.infer<typeof ExplanationSchema>

/////////////////////////////////////////
// EXPLANATION PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationPartialSchema = ExplanationSchema.partial()

export type ExplanationPartial = z.infer<typeof ExplanationPartialSchema>

// EXPLANATION RELATION SCHEMA
//------------------------------------------------------

export type ExplanationRelations = {
  author: UserWithRelations;
  triggeredByUser?: UserWithRelations | null;
  model: ModelWithRelations;
  neuron: NeuronWithRelations;
  votes: VoteWithRelations[];
  type?: ExplanationTypeWithRelations | null;
  explanationModelType?: ExplanationModelTypeWithRelations | null;
  scores: ExplanationScoreWithRelations[];
  activationsV1: ExplanationActivationV1WithRelations[];
};

export type ExplanationWithRelations = z.infer<typeof ExplanationSchema> & ExplanationRelations

export const ExplanationWithRelationsSchema: z.ZodType<ExplanationWithRelations> = ExplanationSchema.merge(z.object({
  author: z.lazy(() => UserWithRelationsSchema),
  triggeredByUser: z.lazy(() => UserWithRelationsSchema).nullable(),
  model: z.lazy(() => ModelWithRelationsSchema),
  neuron: z.lazy(() => NeuronWithRelationsSchema),
  votes: z.lazy(() => VoteWithRelationsSchema).array(),
  type: z.lazy(() => ExplanationTypeWithRelationsSchema).nullable(),
  explanationModelType: z.lazy(() => ExplanationModelTypeWithRelationsSchema).nullable(),
  scores: z.lazy(() => ExplanationScoreWithRelationsSchema).array(),
  activationsV1: z.lazy(() => ExplanationActivationV1WithRelationsSchema).array(),
}))

// EXPLANATION PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationPartialRelations = {
  author?: UserPartialWithRelations;
  triggeredByUser?: UserPartialWithRelations | null;
  model?: ModelPartialWithRelations;
  neuron?: NeuronPartialWithRelations;
  votes?: VotePartialWithRelations[];
  type?: ExplanationTypePartialWithRelations | null;
  explanationModelType?: ExplanationModelTypePartialWithRelations | null;
  scores?: ExplanationScorePartialWithRelations[];
  activationsV1?: ExplanationActivationV1PartialWithRelations[];
};

export type ExplanationPartialWithRelations = z.infer<typeof ExplanationPartialSchema> & ExplanationPartialRelations

export const ExplanationPartialWithRelationsSchema: z.ZodType<ExplanationPartialWithRelations> = ExplanationPartialSchema.merge(z.object({
  author: z.lazy(() => UserPartialWithRelationsSchema),
  triggeredByUser: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  votes: z.lazy(() => VotePartialWithRelationsSchema).array(),
  type: z.lazy(() => ExplanationTypePartialWithRelationsSchema).nullable(),
  explanationModelType: z.lazy(() => ExplanationModelTypePartialWithRelationsSchema).nullable(),
  scores: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
  activationsV1: z.lazy(() => ExplanationActivationV1PartialWithRelationsSchema).array(),
})).partial()

export type ExplanationWithPartialRelations = z.infer<typeof ExplanationSchema> & ExplanationPartialRelations

export const ExplanationWithPartialRelationsSchema: z.ZodType<ExplanationWithPartialRelations> = ExplanationSchema.merge(z.object({
  author: z.lazy(() => UserPartialWithRelationsSchema),
  triggeredByUser: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  votes: z.lazy(() => VotePartialWithRelationsSchema).array(),
  type: z.lazy(() => ExplanationTypePartialWithRelationsSchema).nullable(),
  explanationModelType: z.lazy(() => ExplanationModelTypePartialWithRelationsSchema).nullable(),
  scores: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
  activationsV1: z.lazy(() => ExplanationActivationV1PartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// EXPLANATION MODEL SETTINGS SCHEMA
/////////////////////////////////////////

export const ExplanationModelSettingsSchema = z.object({
  id: z.string().cuid(),
  temperature: z.number().nullable(),
  maxTokens: z.number().int().nullable(),
  topP: z.number().nullable(),
  frequencyPenalty: z.number().nullable(),
})

export type ExplanationModelSettings = z.infer<typeof ExplanationModelSettingsSchema>

/////////////////////////////////////////
// EXPLANATION MODEL SETTINGS PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationModelSettingsPartialSchema = ExplanationModelSettingsSchema.partial()

export type ExplanationModelSettingsPartial = z.infer<typeof ExplanationModelSettingsPartialSchema>

// EXPLANATION MODEL SETTINGS RELATION SCHEMA
//------------------------------------------------------

export type ExplanationModelSettingsRelations = {
  explanationTypes: ExplanationTypeWithRelations[];
};

export type ExplanationModelSettingsWithRelations = z.infer<typeof ExplanationModelSettingsSchema> & ExplanationModelSettingsRelations

export const ExplanationModelSettingsWithRelationsSchema: z.ZodType<ExplanationModelSettingsWithRelations> = ExplanationModelSettingsSchema.merge(z.object({
  explanationTypes: z.lazy(() => ExplanationTypeWithRelationsSchema).array(),
}))

// EXPLANATION MODEL SETTINGS PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationModelSettingsPartialRelations = {
  explanationTypes?: ExplanationTypePartialWithRelations[];
};

export type ExplanationModelSettingsPartialWithRelations = z.infer<typeof ExplanationModelSettingsPartialSchema> & ExplanationModelSettingsPartialRelations

export const ExplanationModelSettingsPartialWithRelationsSchema: z.ZodType<ExplanationModelSettingsPartialWithRelations> = ExplanationModelSettingsPartialSchema.merge(z.object({
  explanationTypes: z.lazy(() => ExplanationTypePartialWithRelationsSchema).array(),
})).partial()

export type ExplanationModelSettingsWithPartialRelations = z.infer<typeof ExplanationModelSettingsSchema> & ExplanationModelSettingsPartialRelations

export const ExplanationModelSettingsWithPartialRelationsSchema: z.ZodType<ExplanationModelSettingsWithPartialRelations> = ExplanationModelSettingsSchema.merge(z.object({
  explanationTypes: z.lazy(() => ExplanationTypePartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// EXPLANATION MODEL TYPE SCHEMA
/////////////////////////////////////////

export const ExplanationModelTypeSchema = z.object({
  name: z.string(),
  displayName: z.string(),
  description: z.string(),
  featured: z.boolean(),
  openRouterModelId: z.string().nullable(),
  url: z.string().nullable(),
  creatorName: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type ExplanationModelType = z.infer<typeof ExplanationModelTypeSchema>

/////////////////////////////////////////
// EXPLANATION MODEL TYPE PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationModelTypePartialSchema = ExplanationModelTypeSchema.partial()

export type ExplanationModelTypePartial = z.infer<typeof ExplanationModelTypePartialSchema>

// EXPLANATION MODEL TYPE RELATION SCHEMA
//------------------------------------------------------

export type ExplanationModelTypeRelations = {
  explanations: ExplanationWithRelations[];
};

export type ExplanationModelTypeWithRelations = z.infer<typeof ExplanationModelTypeSchema> & ExplanationModelTypeRelations

export const ExplanationModelTypeWithRelationsSchema: z.ZodType<ExplanationModelTypeWithRelations> = ExplanationModelTypeSchema.merge(z.object({
  explanations: z.lazy(() => ExplanationWithRelationsSchema).array(),
}))

// EXPLANATION MODEL TYPE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationModelTypePartialRelations = {
  explanations?: ExplanationPartialWithRelations[];
};

export type ExplanationModelTypePartialWithRelations = z.infer<typeof ExplanationModelTypePartialSchema> & ExplanationModelTypePartialRelations

export const ExplanationModelTypePartialWithRelationsSchema: z.ZodType<ExplanationModelTypePartialWithRelations> = ExplanationModelTypePartialSchema.merge(z.object({
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
})).partial()

export type ExplanationModelTypeWithPartialRelations = z.infer<typeof ExplanationModelTypeSchema> & ExplanationModelTypePartialRelations

export const ExplanationModelTypeWithPartialRelationsSchema: z.ZodType<ExplanationModelTypeWithPartialRelations> = ExplanationModelTypeSchema.merge(z.object({
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// EXPLANATION TYPE SCHEMA
/////////////////////////////////////////

export const ExplanationTypeSchema = z.object({
  name: z.string(),
  displayName: z.string(),
  description: z.string(),
  featured: z.boolean(),
  isAttention: z.boolean(),
  explainerModelSettingsId: z.string().nullable(),
  settings: z.string().nullable(),
  url: z.string().nullable(),
  creatorName: z.string(),
  creatorId: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type ExplanationType = z.infer<typeof ExplanationTypeSchema>

/////////////////////////////////////////
// EXPLANATION TYPE PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationTypePartialSchema = ExplanationTypeSchema.partial()

export type ExplanationTypePartial = z.infer<typeof ExplanationTypePartialSchema>

// EXPLANATION TYPE RELATION SCHEMA
//------------------------------------------------------

export type ExplanationTypeRelations = {
  explainerModelSettings?: ExplanationModelSettingsWithRelations | null;
  creator: UserWithRelations;
  explanations: ExplanationWithRelations[];
};

export type ExplanationTypeWithRelations = z.infer<typeof ExplanationTypeSchema> & ExplanationTypeRelations

export const ExplanationTypeWithRelationsSchema: z.ZodType<ExplanationTypeWithRelations> = ExplanationTypeSchema.merge(z.object({
  explainerModelSettings: z.lazy(() => ExplanationModelSettingsWithRelationsSchema).nullable(),
  creator: z.lazy(() => UserWithRelationsSchema),
  explanations: z.lazy(() => ExplanationWithRelationsSchema).array(),
}))

// EXPLANATION TYPE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationTypePartialRelations = {
  explainerModelSettings?: ExplanationModelSettingsPartialWithRelations | null;
  creator?: UserPartialWithRelations;
  explanations?: ExplanationPartialWithRelations[];
};

export type ExplanationTypePartialWithRelations = z.infer<typeof ExplanationTypePartialSchema> & ExplanationTypePartialRelations

export const ExplanationTypePartialWithRelationsSchema: z.ZodType<ExplanationTypePartialWithRelations> = ExplanationTypePartialSchema.merge(z.object({
  explainerModelSettings: z.lazy(() => ExplanationModelSettingsPartialWithRelationsSchema).nullable(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
})).partial()

export type ExplanationTypeWithPartialRelations = z.infer<typeof ExplanationTypeSchema> & ExplanationTypePartialRelations

export const ExplanationTypeWithPartialRelationsSchema: z.ZodType<ExplanationTypeWithPartialRelations> = ExplanationTypeSchema.merge(z.object({
  explainerModelSettings: z.lazy(() => ExplanationModelSettingsPartialWithRelationsSchema).nullable(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  explanations: z.lazy(() => ExplanationPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// EXPLANATION SCORE SCHEMA
/////////////////////////////////////////

export const ExplanationScoreSchema = z.object({
  id: z.string().cuid(),
  initiatedByUserId: z.string(),
  value: z.number(),
  explanationId: z.string(),
  explanationScoreTypeName: z.string(),
  explanationScoreModelName: z.string(),
  jsonDetails: z.string(),
  createdAt: z.coerce.date(),
})

export type ExplanationScore = z.infer<typeof ExplanationScoreSchema>

/////////////////////////////////////////
// EXPLANATION SCORE PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationScorePartialSchema = ExplanationScoreSchema.partial()

export type ExplanationScorePartial = z.infer<typeof ExplanationScorePartialSchema>

// EXPLANATION SCORE RELATION SCHEMA
//------------------------------------------------------

export type ExplanationScoreRelations = {
  initiatedByUser: UserWithRelations;
  explanation: ExplanationWithRelations;
  explanationScoreType: ExplanationScoreTypeWithRelations;
  explanationScoreModel: ExplanationScoreModelWithRelations;
};

export type ExplanationScoreWithRelations = z.infer<typeof ExplanationScoreSchema> & ExplanationScoreRelations

export const ExplanationScoreWithRelationsSchema: z.ZodType<ExplanationScoreWithRelations> = ExplanationScoreSchema.merge(z.object({
  initiatedByUser: z.lazy(() => UserWithRelationsSchema),
  explanation: z.lazy(() => ExplanationWithRelationsSchema),
  explanationScoreType: z.lazy(() => ExplanationScoreTypeWithRelationsSchema),
  explanationScoreModel: z.lazy(() => ExplanationScoreModelWithRelationsSchema),
}))

// EXPLANATION SCORE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationScorePartialRelations = {
  initiatedByUser?: UserPartialWithRelations;
  explanation?: ExplanationPartialWithRelations;
  explanationScoreType?: ExplanationScoreTypePartialWithRelations;
  explanationScoreModel?: ExplanationScoreModelPartialWithRelations;
};

export type ExplanationScorePartialWithRelations = z.infer<typeof ExplanationScorePartialSchema> & ExplanationScorePartialRelations

export const ExplanationScorePartialWithRelationsSchema: z.ZodType<ExplanationScorePartialWithRelations> = ExplanationScorePartialSchema.merge(z.object({
  initiatedByUser: z.lazy(() => UserPartialWithRelationsSchema),
  explanation: z.lazy(() => ExplanationPartialWithRelationsSchema),
  explanationScoreType: z.lazy(() => ExplanationScoreTypePartialWithRelationsSchema),
  explanationScoreModel: z.lazy(() => ExplanationScoreModelPartialWithRelationsSchema),
})).partial()

export type ExplanationScoreWithPartialRelations = z.infer<typeof ExplanationScoreSchema> & ExplanationScorePartialRelations

export const ExplanationScoreWithPartialRelationsSchema: z.ZodType<ExplanationScoreWithPartialRelations> = ExplanationScoreSchema.merge(z.object({
  initiatedByUser: z.lazy(() => UserPartialWithRelationsSchema),
  explanation: z.lazy(() => ExplanationPartialWithRelationsSchema),
  explanationScoreType: z.lazy(() => ExplanationScoreTypePartialWithRelationsSchema),
  explanationScoreModel: z.lazy(() => ExplanationScoreModelPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// EXPLANATION SCORE TYPE SCHEMA
/////////////////////////////////////////

export const ExplanationScoreTypeSchema = z.object({
  name: z.string(),
  displayName: z.string(),
  description: z.string(),
  featured: z.boolean(),
  isAttention: z.boolean(),
  scoreDescription: z.string().nullable(),
  scorerModel: z.string().nullable(),
  settings: z.string().nullable(),
  url: z.string().nullable(),
  creatorName: z.string(),
  creatorId: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type ExplanationScoreType = z.infer<typeof ExplanationScoreTypeSchema>

/////////////////////////////////////////
// EXPLANATION SCORE TYPE PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationScoreTypePartialSchema = ExplanationScoreTypeSchema.partial()

export type ExplanationScoreTypePartial = z.infer<typeof ExplanationScoreTypePartialSchema>

// EXPLANATION SCORE TYPE RELATION SCHEMA
//------------------------------------------------------

export type ExplanationScoreTypeRelations = {
  explanationScores: ExplanationScoreWithRelations[];
  creator: UserWithRelations;
};

export type ExplanationScoreTypeWithRelations = z.infer<typeof ExplanationScoreTypeSchema> & ExplanationScoreTypeRelations

export const ExplanationScoreTypeWithRelationsSchema: z.ZodType<ExplanationScoreTypeWithRelations> = ExplanationScoreTypeSchema.merge(z.object({
  explanationScores: z.lazy(() => ExplanationScoreWithRelationsSchema).array(),
  creator: z.lazy(() => UserWithRelationsSchema),
}))

// EXPLANATION SCORE TYPE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationScoreTypePartialRelations = {
  explanationScores?: ExplanationScorePartialWithRelations[];
  creator?: UserPartialWithRelations;
};

export type ExplanationScoreTypePartialWithRelations = z.infer<typeof ExplanationScoreTypePartialSchema> & ExplanationScoreTypePartialRelations

export const ExplanationScoreTypePartialWithRelationsSchema: z.ZodType<ExplanationScoreTypePartialWithRelations> = ExplanationScoreTypePartialSchema.merge(z.object({
  explanationScores: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type ExplanationScoreTypeWithPartialRelations = z.infer<typeof ExplanationScoreTypeSchema> & ExplanationScoreTypePartialRelations

export const ExplanationScoreTypeWithPartialRelationsSchema: z.ZodType<ExplanationScoreTypeWithPartialRelations> = ExplanationScoreTypeSchema.merge(z.object({
  explanationScores: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// EXPLANATION SCORE MODEL SCHEMA
/////////////////////////////////////////

export const ExplanationScoreModelSchema = z.object({
  name: z.string(),
  displayName: z.string(),
  description: z.string(),
  featured: z.boolean(),
  openRouterModelId: z.string().nullable(),
  url: z.string().nullable(),
  creatorName: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type ExplanationScoreModel = z.infer<typeof ExplanationScoreModelSchema>

/////////////////////////////////////////
// EXPLANATION SCORE MODEL PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationScoreModelPartialSchema = ExplanationScoreModelSchema.partial()

export type ExplanationScoreModelPartial = z.infer<typeof ExplanationScoreModelPartialSchema>

// EXPLANATION SCORE MODEL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationScoreModelRelations = {
  explanationScores: ExplanationScoreWithRelations[];
};

export type ExplanationScoreModelWithRelations = z.infer<typeof ExplanationScoreModelSchema> & ExplanationScoreModelRelations

export const ExplanationScoreModelWithRelationsSchema: z.ZodType<ExplanationScoreModelWithRelations> = ExplanationScoreModelSchema.merge(z.object({
  explanationScores: z.lazy(() => ExplanationScoreWithRelationsSchema).array(),
}))

// EXPLANATION SCORE MODEL PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationScoreModelPartialRelations = {
  explanationScores?: ExplanationScorePartialWithRelations[];
};

export type ExplanationScoreModelPartialWithRelations = z.infer<typeof ExplanationScoreModelPartialSchema> & ExplanationScoreModelPartialRelations

export const ExplanationScoreModelPartialWithRelationsSchema: z.ZodType<ExplanationScoreModelPartialWithRelations> = ExplanationScoreModelPartialSchema.merge(z.object({
  explanationScores: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
})).partial()

export type ExplanationScoreModelWithPartialRelations = z.infer<typeof ExplanationScoreModelSchema> & ExplanationScoreModelPartialRelations

export const ExplanationScoreModelWithPartialRelationsSchema: z.ZodType<ExplanationScoreModelWithPartialRelations> = ExplanationScoreModelSchema.merge(z.object({
  explanationScores: z.lazy(() => ExplanationScorePartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// EXPLANATION ACTIVATION V 1 SCHEMA
/////////////////////////////////////////

export const ExplanationActivationV1Schema = z.object({
  id: z.string().cuid(),
  explanationId: z.string(),
  activationId: z.string(),
  expectedValues: z.number().array(),
  score: z.number(),
  scorerId: z.string().nullable(),
  scorerAutoInterpModel: z.string().nullable(),
  version: z.number().int(),
  createdAt: z.coerce.date(),
})

export type ExplanationActivationV1 = z.infer<typeof ExplanationActivationV1Schema>

/////////////////////////////////////////
// EXPLANATION ACTIVATION V 1 PARTIAL SCHEMA
/////////////////////////////////////////

export const ExplanationActivationV1PartialSchema = ExplanationActivationV1Schema.partial()

export type ExplanationActivationV1Partial = z.infer<typeof ExplanationActivationV1PartialSchema>

// EXPLANATION ACTIVATION V 1 RELATION SCHEMA
//------------------------------------------------------

export type ExplanationActivationV1Relations = {
  explanation: ExplanationWithRelations;
  activation: ActivationWithRelations;
  scorer?: UserWithRelations | null;
};

export type ExplanationActivationV1WithRelations = z.infer<typeof ExplanationActivationV1Schema> & ExplanationActivationV1Relations

export const ExplanationActivationV1WithRelationsSchema: z.ZodType<ExplanationActivationV1WithRelations> = ExplanationActivationV1Schema.merge(z.object({
  explanation: z.lazy(() => ExplanationWithRelationsSchema),
  activation: z.lazy(() => ActivationWithRelationsSchema),
  scorer: z.lazy(() => UserWithRelationsSchema).nullable(),
}))

// EXPLANATION ACTIVATION V 1 PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ExplanationActivationV1PartialRelations = {
  explanation?: ExplanationPartialWithRelations;
  activation?: ActivationPartialWithRelations;
  scorer?: UserPartialWithRelations | null;
};

export type ExplanationActivationV1PartialWithRelations = z.infer<typeof ExplanationActivationV1PartialSchema> & ExplanationActivationV1PartialRelations

export const ExplanationActivationV1PartialWithRelationsSchema: z.ZodType<ExplanationActivationV1PartialWithRelations> = ExplanationActivationV1PartialSchema.merge(z.object({
  explanation: z.lazy(() => ExplanationPartialWithRelationsSchema),
  activation: z.lazy(() => ActivationPartialWithRelationsSchema),
  scorer: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
})).partial()

export type ExplanationActivationV1WithPartialRelations = z.infer<typeof ExplanationActivationV1Schema> & ExplanationActivationV1PartialRelations

export const ExplanationActivationV1WithPartialRelationsSchema: z.ZodType<ExplanationActivationV1WithPartialRelations> = ExplanationActivationV1Schema.merge(z.object({
  explanation: z.lazy(() => ExplanationPartialWithRelationsSchema),
  activation: z.lazy(() => ActivationPartialWithRelationsSchema),
  scorer: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
}).partial())

/////////////////////////////////////////
// VOTE SCHEMA
/////////////////////////////////////////

export const VoteSchema = z.object({
  id: z.string().cuid(),
  voterId: z.string(),
  reason: z.string().nullable(),
  points: z.number().int(),
  explanationId: z.string(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
})

export type Vote = z.infer<typeof VoteSchema>

/////////////////////////////////////////
// VOTE PARTIAL SCHEMA
/////////////////////////////////////////

export const VotePartialSchema = VoteSchema.partial()

export type VotePartial = z.infer<typeof VotePartialSchema>

// VOTE RELATION SCHEMA
//------------------------------------------------------

export type VoteRelations = {
  explanation: ExplanationWithRelations;
  voter: UserWithRelations;
};

export type VoteWithRelations = z.infer<typeof VoteSchema> & VoteRelations

export const VoteWithRelationsSchema: z.ZodType<VoteWithRelations> = VoteSchema.merge(z.object({
  explanation: z.lazy(() => ExplanationWithRelationsSchema),
  voter: z.lazy(() => UserWithRelationsSchema),
}))

// VOTE PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type VotePartialRelations = {
  explanation?: ExplanationPartialWithRelations;
  voter?: UserPartialWithRelations;
};

export type VotePartialWithRelations = z.infer<typeof VotePartialSchema> & VotePartialRelations

export const VotePartialWithRelationsSchema: z.ZodType<VotePartialWithRelations> = VotePartialSchema.merge(z.object({
  explanation: z.lazy(() => ExplanationPartialWithRelationsSchema),
  voter: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type VoteWithPartialRelations = z.infer<typeof VoteSchema> & VotePartialRelations

export const VoteWithPartialRelationsSchema: z.ZodType<VoteWithPartialRelations> = VoteSchema.merge(z.object({
  explanation: z.lazy(() => ExplanationPartialWithRelationsSchema),
  voter: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// ACTIVATION SCHEMA
/////////////////////////////////////////

export const ActivationSchema = z.object({
  id: z.string().cuid(),
  tokens: z.string().array(),
  dataIndex: z.string().nullable(),
  index: z.string(),
  layer: z.string(),
  modelId: z.string(),
  dataSource: z.string().nullable(),
  maxValue: z.number(),
  maxValueTokenIndex: z.number().int(),
  minValue: z.number(),
  values: z.number().array(),
  dfaValues: z.number().array(),
  dfaTargetIndex: z.number().int().nullable(),
  dfaMaxValue: z.number().nullable(),
  creatorId: z.string(),
  createdAt: z.coerce.date(),
  lossValues: z.number().array(),
  logitContributions: z.string().nullable(),
  binMin: z.number().nullable(),
  binMax: z.number().nullable(),
  binContains: z.number().nullable(),
  qualifyingTokenIndex: z.number().int().nullable(),
})

export type Activation = z.infer<typeof ActivationSchema>

/////////////////////////////////////////
// ACTIVATION PARTIAL SCHEMA
/////////////////////////////////////////

export const ActivationPartialSchema = ActivationSchema.partial()

export type ActivationPartial = z.infer<typeof ActivationPartialSchema>

// ACTIVATION RELATION SCHEMA
//------------------------------------------------------

export type ActivationRelations = {
  neuron: NeuronWithRelations;
  explanations: ExplanationActivationV1WithRelations[];
  savedSearch: SavedSearchActivationWithRelations[];
  creator: UserWithRelations;
  lists: ListsOnActivationsWithRelations[];
};

export type ActivationWithRelations = z.infer<typeof ActivationSchema> & ActivationRelations

export const ActivationWithRelationsSchema: z.ZodType<ActivationWithRelations> = ActivationSchema.merge(z.object({
  neuron: z.lazy(() => NeuronWithRelationsSchema),
  explanations: z.lazy(() => ExplanationActivationV1WithRelationsSchema).array(),
  savedSearch: z.lazy(() => SavedSearchActivationWithRelationsSchema).array(),
  creator: z.lazy(() => UserWithRelationsSchema),
  lists: z.lazy(() => ListsOnActivationsWithRelationsSchema).array(),
}))

// ACTIVATION PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type ActivationPartialRelations = {
  neuron?: NeuronPartialWithRelations;
  explanations?: ExplanationActivationV1PartialWithRelations[];
  savedSearch?: SavedSearchActivationPartialWithRelations[];
  creator?: UserPartialWithRelations;
  lists?: ListsOnActivationsPartialWithRelations[];
};

export type ActivationPartialWithRelations = z.infer<typeof ActivationPartialSchema> & ActivationPartialRelations

export const ActivationPartialWithRelationsSchema: z.ZodType<ActivationPartialWithRelations> = ActivationPartialSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  explanations: z.lazy(() => ExplanationActivationV1PartialWithRelationsSchema).array(),
  savedSearch: z.lazy(() => SavedSearchActivationPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  lists: z.lazy(() => ListsOnActivationsPartialWithRelationsSchema).array(),
})).partial()

export type ActivationWithPartialRelations = z.infer<typeof ActivationSchema> & ActivationPartialRelations

export const ActivationWithPartialRelationsSchema: z.ZodType<ActivationWithPartialRelations> = ActivationSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  explanations: z.lazy(() => ExplanationActivationV1PartialWithRelationsSchema).array(),
  savedSearch: z.lazy(() => SavedSearchActivationPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema),
  lists: z.lazy(() => ListsOnActivationsPartialWithRelationsSchema).array(),
}).partial())

/////////////////////////////////////////
// COMMENT SCHEMA
/////////////////////////////////////////

export const CommentSchema = z.object({
  id: z.string().cuid(),
  modelId: z.string(),
  layer: z.string(),
  index: z.string(),
  text: z.string(),
  userId: z.string(),
  createdAt: z.coerce.date(),
})

export type Comment = z.infer<typeof CommentSchema>

/////////////////////////////////////////
// COMMENT PARTIAL SCHEMA
/////////////////////////////////////////

export const CommentPartialSchema = CommentSchema.partial()

export type CommentPartial = z.infer<typeof CommentPartialSchema>

// COMMENT RELATION SCHEMA
//------------------------------------------------------

export type CommentRelations = {
  neuron: NeuronWithRelations;
  user: UserWithRelations;
};

export type CommentWithRelations = z.infer<typeof CommentSchema> & CommentRelations

export const CommentWithRelationsSchema: z.ZodType<CommentWithRelations> = CommentSchema.merge(z.object({
  neuron: z.lazy(() => NeuronWithRelationsSchema),
  user: z.lazy(() => UserWithRelationsSchema),
}))

// COMMENT PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type CommentPartialRelations = {
  neuron?: NeuronPartialWithRelations;
  user?: UserPartialWithRelations;
};

export type CommentPartialWithRelations = z.infer<typeof CommentPartialSchema> & CommentPartialRelations

export const CommentPartialWithRelationsSchema: z.ZodType<CommentPartialWithRelations> = CommentPartialSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type CommentWithPartialRelations = z.infer<typeof CommentSchema> & CommentPartialRelations

export const CommentWithPartialRelationsSchema: z.ZodType<CommentWithPartialRelations> = CommentSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// BOOKMARK SCHEMA
/////////////////////////////////////////

export const BookmarkSchema = z.object({
  id: z.string().cuid(),
  modelId: z.string(),
  layer: z.string(),
  index: z.string(),
  userId: z.string(),
  createdAt: z.coerce.date(),
})

export type Bookmark = z.infer<typeof BookmarkSchema>

/////////////////////////////////////////
// BOOKMARK PARTIAL SCHEMA
/////////////////////////////////////////

export const BookmarkPartialSchema = BookmarkSchema.partial()

export type BookmarkPartial = z.infer<typeof BookmarkPartialSchema>

// BOOKMARK RELATION SCHEMA
//------------------------------------------------------

export type BookmarkRelations = {
  neuron: NeuronWithRelations;
  user: UserWithRelations;
};

export type BookmarkWithRelations = z.infer<typeof BookmarkSchema> & BookmarkRelations

export const BookmarkWithRelationsSchema: z.ZodType<BookmarkWithRelations> = BookmarkSchema.merge(z.object({
  neuron: z.lazy(() => NeuronWithRelationsSchema),
  user: z.lazy(() => UserWithRelationsSchema),
}))

// BOOKMARK PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type BookmarkPartialRelations = {
  neuron?: NeuronPartialWithRelations;
  user?: UserPartialWithRelations;
};

export type BookmarkPartialWithRelations = z.infer<typeof BookmarkPartialSchema> & BookmarkPartialRelations

export const BookmarkPartialWithRelationsSchema: z.ZodType<BookmarkPartialWithRelations> = BookmarkPartialSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
})).partial()

export type BookmarkWithPartialRelations = z.infer<typeof BookmarkSchema> & BookmarkPartialRelations

export const BookmarkWithPartialRelationsSchema: z.ZodType<BookmarkWithPartialRelations> = BookmarkSchema.merge(z.object({
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
  user: z.lazy(() => UserPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// SAVED SEARCH SCHEMA
/////////////////////////////////////////

export const SavedSearchSchema = z.object({
  id: z.string().cuid(),
  query: z.string(),
  tokens: z.string().array(),
  modelId: z.string(),
  selectedLayers: z.string().array(),
  sortByIndexes: z.number().int().array(),
  ignoreBos: z.boolean(),
  sourceSet: z.string().nullable(),
  userId: z.string().nullable(),
  updatedAt: z.coerce.date(),
  createdAt: z.coerce.date(),
  counts: z.string(),
  numResults: z.number().int(),
  densityThreshold: z.number(),
})

export type SavedSearch = z.infer<typeof SavedSearchSchema>

/////////////////////////////////////////
// SAVED SEARCH PARTIAL SCHEMA
/////////////////////////////////////////

export const SavedSearchPartialSchema = SavedSearchSchema.partial()

export type SavedSearchPartial = z.infer<typeof SavedSearchPartialSchema>

// SAVED SEARCH RELATION SCHEMA
//------------------------------------------------------

export type SavedSearchRelations = {
  model: ModelWithRelations;
  activations: SavedSearchActivationWithRelations[];
  user?: UserWithRelations | null;
};

export type SavedSearchWithRelations = z.infer<typeof SavedSearchSchema> & SavedSearchRelations

export const SavedSearchWithRelationsSchema: z.ZodType<SavedSearchWithRelations> = SavedSearchSchema.merge(z.object({
  model: z.lazy(() => ModelWithRelationsSchema),
  activations: z.lazy(() => SavedSearchActivationWithRelationsSchema).array(),
  user: z.lazy(() => UserWithRelationsSchema).nullable(),
}))

// SAVED SEARCH PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SavedSearchPartialRelations = {
  model?: ModelPartialWithRelations;
  activations?: SavedSearchActivationPartialWithRelations[];
  user?: UserPartialWithRelations | null;
};

export type SavedSearchPartialWithRelations = z.infer<typeof SavedSearchPartialSchema> & SavedSearchPartialRelations

export const SavedSearchPartialWithRelationsSchema: z.ZodType<SavedSearchPartialWithRelations> = SavedSearchPartialSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  activations: z.lazy(() => SavedSearchActivationPartialWithRelationsSchema).array(),
  user: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
})).partial()

export type SavedSearchWithPartialRelations = z.infer<typeof SavedSearchSchema> & SavedSearchPartialRelations

export const SavedSearchWithPartialRelationsSchema: z.ZodType<SavedSearchWithPartialRelations> = SavedSearchSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  activations: z.lazy(() => SavedSearchActivationPartialWithRelationsSchema).array(),
  user: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
}).partial())

/////////////////////////////////////////
// SAVED SEARCH ACTIVATION SCHEMA
/////////////////////////////////////////

export const SavedSearchActivationSchema = z.object({
  savedSearchId: z.string(),
  activationId: z.string(),
  order: z.number().int(),
})

export type SavedSearchActivation = z.infer<typeof SavedSearchActivationSchema>

/////////////////////////////////////////
// SAVED SEARCH ACTIVATION PARTIAL SCHEMA
/////////////////////////////////////////

export const SavedSearchActivationPartialSchema = SavedSearchActivationSchema.partial()

export type SavedSearchActivationPartial = z.infer<typeof SavedSearchActivationPartialSchema>

// SAVED SEARCH ACTIVATION RELATION SCHEMA
//------------------------------------------------------

export type SavedSearchActivationRelations = {
  savedSearch: SavedSearchWithRelations;
  activation: ActivationWithRelations;
};

export type SavedSearchActivationWithRelations = z.infer<typeof SavedSearchActivationSchema> & SavedSearchActivationRelations

export const SavedSearchActivationWithRelationsSchema: z.ZodType<SavedSearchActivationWithRelations> = SavedSearchActivationSchema.merge(z.object({
  savedSearch: z.lazy(() => SavedSearchWithRelationsSchema),
  activation: z.lazy(() => ActivationWithRelationsSchema),
}))

// SAVED SEARCH ACTIVATION PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SavedSearchActivationPartialRelations = {
  savedSearch?: SavedSearchPartialWithRelations;
  activation?: ActivationPartialWithRelations;
};

export type SavedSearchActivationPartialWithRelations = z.infer<typeof SavedSearchActivationPartialSchema> & SavedSearchActivationPartialRelations

export const SavedSearchActivationPartialWithRelationsSchema: z.ZodType<SavedSearchActivationPartialWithRelations> = SavedSearchActivationPartialSchema.merge(z.object({
  savedSearch: z.lazy(() => SavedSearchPartialWithRelationsSchema),
  activation: z.lazy(() => ActivationPartialWithRelationsSchema),
})).partial()

export type SavedSearchActivationWithPartialRelations = z.infer<typeof SavedSearchActivationSchema> & SavedSearchActivationPartialRelations

export const SavedSearchActivationWithPartialRelationsSchema: z.ZodType<SavedSearchActivationWithPartialRelations> = SavedSearchActivationSchema.merge(z.object({
  savedSearch: z.lazy(() => SavedSearchPartialWithRelationsSchema),
  activation: z.lazy(() => ActivationPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// STEER OUTPUT TO NEURON SCHEMA
/////////////////////////////////////////

export const SteerOutputToNeuronSchema = z.object({
  modelId: z.string(),
  layer: z.string(),
  index: z.string(),
  strength: z.number(),
  steerOutputId: z.string(),
})

export type SteerOutputToNeuron = z.infer<typeof SteerOutputToNeuronSchema>

/////////////////////////////////////////
// STEER OUTPUT TO NEURON PARTIAL SCHEMA
/////////////////////////////////////////

export const SteerOutputToNeuronPartialSchema = SteerOutputToNeuronSchema.partial()

export type SteerOutputToNeuronPartial = z.infer<typeof SteerOutputToNeuronPartialSchema>

// STEER OUTPUT TO NEURON RELATION SCHEMA
//------------------------------------------------------

export type SteerOutputToNeuronRelations = {
  steerOutput: SteerOutputWithRelations;
  neuron: NeuronWithRelations;
};

export type SteerOutputToNeuronWithRelations = z.infer<typeof SteerOutputToNeuronSchema> & SteerOutputToNeuronRelations

export const SteerOutputToNeuronWithRelationsSchema: z.ZodType<SteerOutputToNeuronWithRelations> = SteerOutputToNeuronSchema.merge(z.object({
  steerOutput: z.lazy(() => SteerOutputWithRelationsSchema),
  neuron: z.lazy(() => NeuronWithRelationsSchema),
}))

// STEER OUTPUT TO NEURON PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SteerOutputToNeuronPartialRelations = {
  steerOutput?: SteerOutputPartialWithRelations;
  neuron?: NeuronPartialWithRelations;
};

export type SteerOutputToNeuronPartialWithRelations = z.infer<typeof SteerOutputToNeuronPartialSchema> & SteerOutputToNeuronPartialRelations

export const SteerOutputToNeuronPartialWithRelationsSchema: z.ZodType<SteerOutputToNeuronPartialWithRelations> = SteerOutputToNeuronPartialSchema.merge(z.object({
  steerOutput: z.lazy(() => SteerOutputPartialWithRelationsSchema),
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
})).partial()

export type SteerOutputToNeuronWithPartialRelations = z.infer<typeof SteerOutputToNeuronSchema> & SteerOutputToNeuronPartialRelations

export const SteerOutputToNeuronWithPartialRelationsSchema: z.ZodType<SteerOutputToNeuronWithPartialRelations> = SteerOutputToNeuronSchema.merge(z.object({
  steerOutput: z.lazy(() => SteerOutputPartialWithRelationsSchema),
  neuron: z.lazy(() => NeuronPartialWithRelationsSchema),
}).partial())

/////////////////////////////////////////
// STEER OUTPUT SCHEMA
/////////////////////////////////////////

export const SteerOutputSchema = z.object({
  type: SteerOutputTypeSchema,
  id: z.string().cuid(),
  modelId: z.string(),
  steerSpecialTokens: z.boolean(),
  inputText: z.string(),
  inputTextChatTemplate: z.string().nullable(),
  outputText: z.string(),
  outputTextChatTemplate: z.string().nullable(),
  temperature: z.number(),
  numTokens: z.number().int(),
  freqPenalty: z.number(),
  seed: z.number(),
  strengthMultiplier: z.number(),
  steerMethod: z.string(),
  createdAt: z.coerce.date(),
  creatorId: z.string().nullable(),
  version: z.number().int(),
  connectedDefaultOutputId: z.string().nullable(),
  connectedSteerOutputIds: z.string().array(),
})

export type SteerOutput = z.infer<typeof SteerOutputSchema>

/////////////////////////////////////////
// STEER OUTPUT PARTIAL SCHEMA
/////////////////////////////////////////

export const SteerOutputPartialSchema = SteerOutputSchema.partial()

export type SteerOutputPartial = z.infer<typeof SteerOutputPartialSchema>

// STEER OUTPUT RELATION SCHEMA
//------------------------------------------------------

export type SteerOutputRelations = {
  model: ModelWithRelations;
  toNeurons: SteerOutputToNeuronWithRelations[];
  creator?: UserWithRelations | null;
  connectedDefaultOutput?: SteerOutputWithRelations | null;
  connectedSteerOutputs: SteerOutputWithRelations[];
};

export type SteerOutputWithRelations = z.infer<typeof SteerOutputSchema> & SteerOutputRelations

export const SteerOutputWithRelationsSchema: z.ZodType<SteerOutputWithRelations> = SteerOutputSchema.merge(z.object({
  model: z.lazy(() => ModelWithRelationsSchema),
  toNeurons: z.lazy(() => SteerOutputToNeuronWithRelationsSchema).array(),
  creator: z.lazy(() => UserWithRelationsSchema).nullable(),
  connectedDefaultOutput: z.lazy(() => SteerOutputWithRelationsSchema).nullable(),
  connectedSteerOutputs: z.lazy(() => SteerOutputWithRelationsSchema).array(),
}))

// STEER OUTPUT PARTIAL RELATION SCHEMA
//------------------------------------------------------

export type SteerOutputPartialRelations = {
  model?: ModelPartialWithRelations;
  toNeurons?: SteerOutputToNeuronPartialWithRelations[];
  creator?: UserPartialWithRelations | null;
  connectedDefaultOutput?: SteerOutputPartialWithRelations | null;
  connectedSteerOutputs?: SteerOutputPartialWithRelations[];
};

export type SteerOutputPartialWithRelations = z.infer<typeof SteerOutputPartialSchema> & SteerOutputPartialRelations

export const SteerOutputPartialWithRelationsSchema: z.ZodType<SteerOutputPartialWithRelations> = SteerOutputPartialSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  toNeurons: z.lazy(() => SteerOutputToNeuronPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
  connectedDefaultOutput: z.lazy(() => SteerOutputPartialWithRelationsSchema).nullable(),
  connectedSteerOutputs: z.lazy(() => SteerOutputPartialWithRelationsSchema).array(),
})).partial()

export type SteerOutputWithPartialRelations = z.infer<typeof SteerOutputSchema> & SteerOutputPartialRelations

export const SteerOutputWithPartialRelationsSchema: z.ZodType<SteerOutputWithPartialRelations> = SteerOutputSchema.merge(z.object({
  model: z.lazy(() => ModelPartialWithRelationsSchema),
  toNeurons: z.lazy(() => SteerOutputToNeuronPartialWithRelationsSchema).array(),
  creator: z.lazy(() => UserPartialWithRelationsSchema).nullable(),
  connectedDefaultOutput: z.lazy(() => SteerOutputPartialWithRelationsSchema).nullable(),
  connectedSteerOutputs: z.lazy(() => SteerOutputPartialWithRelationsSchema).array(),
}).partial())
