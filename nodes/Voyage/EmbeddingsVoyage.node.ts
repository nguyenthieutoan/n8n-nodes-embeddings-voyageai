import {
	NodeConnectionTypes,
	NodeApiError,
	type INodeProperties,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';
import type { EmbeddingsParams } from '@langchain/core/embeddings';

/* eslint-disable @typescript-eslint/no-var-requires */
function requireN8nDependency(dependencyName: string): any {
	// 1. Try normal require first
	try { return require(dependencyName); } catch (_) {}

	// 2. Resolve relative to require.main (n8n itself)
	if (require.main && require.main.paths) {
		try {
			const p = require.resolve(dependencyName, { paths: require.main.paths });
			return require(p);
		} catch (_) {}
	}

	// 3. Fallback: resolve from n8n-workflow path without importing path or fs
	try {
		const workflowResolve = require.resolve('n8n-workflow');
		const index = workflowResolve.indexOf('node_modules');
		if (index !== -1) {
			const base = workflowResolve.substring(0, index + 12);
			return require(base + '/' + dependencyName);
		}
	} catch (_) {}

	throw new Error(`Could not resolve ${dependencyName} from n8n's runtime`);
}

function getAiUtilities(): any {
	try {
		const dep = ['@n8n', 'ai-utilities'].join('/');
		return requireN8nDependency(dep);
	} catch (e) {
		return {
			getConnectionHintNoticeField: (hints: any) => ({
				displayName: '',
				name: 'notice',
				type: 'notice',
				default: '',
			}),
			logWrapper: (instance: any, context: any) => instance,
		};
	}
}

abstract class LocalEmbeddings {
	lc_namespace = ['langchain', 'embeddings'];
	constructor(fields: any) {}
	abstract embedDocuments(documents: string[]): Promise<number[][]>;
	abstract embedQuery(document: string): Promise<number[]>;
}

// Khai báo cấu trúc các tham số khởi tạo mô hình nhúng Voyage
export interface VoyageEmbeddingsParams extends EmbeddingsParams {
	apiKey: string;
	baseUrl: string;
	modelName: string;
	inputType?: string;
	outputDimension?: number;
	outputDtype?: string;
	truncation?: boolean;
	encodingFormat?: string;
}

// Lớp Wrapper tuân thủ chuẩn giao tiếp toán học của LangChain trong n8n
export class VoyageEmbeddingsModel extends LocalEmbeddings {
	private apiKey: string;
	private baseUrl: string;
	private modelName: string;
	private inputType?: string;
	private outputDimension?: number;
	private outputDtype?: string;
	private truncation?: boolean;
	private encodingFormat?: string;
	private parentContext: ISupplyDataFunctions;

	constructor(fields: VoyageEmbeddingsParams, parentContext: ISupplyDataFunctions) {
		super(fields);
		patchVoyageEmbeddingsPrototype();
		this.apiKey = fields.apiKey;
		this.baseUrl = fields.baseUrl || 'https://api.voyageai.com/v1';
		this.modelName = fields.modelName;
		this.inputType = fields.inputType || undefined;
		this.outputDimension = fields.outputDimension;
		this.outputDtype = fields.outputDtype || 'float';
		this.truncation = fields.truncation;
		this.encodingFormat = fields.encodingFormat || undefined;
		this.parentContext = parentContext;
	}

	// Giao tiếp HTTP trực tiếp với API của Voyage AI qua helper n8n
	private async generateEmbeddings(texts: string[]): Promise<number[][]> {
		const payload: Record<string, any> = {
			input: texts,
			model: this.modelName,
			output_dtype: this.outputDtype,
		};

		if (this.inputType) {
			payload.input_type = this.inputType;
		}
		if (this.outputDimension) {
			payload.output_dimension = this.outputDimension;
		}
		if (this.truncation !== undefined) {
			payload.truncation = this.truncation;
		}
		if (this.encodingFormat) {
			payload.encoding_format = this.encodingFormat;
		}

		let jsonResponse: any;

		try {
			jsonResponse = await this.parentContext.helpers.httpRequest({
				method: 'POST',
				url: `${this.baseUrl}/embeddings`,
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${this.apiKey}`,
				},
				body: payload,
				json: true,
			});
		} catch (error) {
			throw new NodeApiError(this.parentContext.getNode(), error as any);
		}

		if (!jsonResponse || !jsonResponse.data || !Array.isArray(jsonResponse.data)) {
			throw new NodeApiError(this.parentContext.getNode(), jsonResponse as any, {
				message: 'Invalid data format returned from Voyage AI API.',
			});
		}

		// Đảm bảo dữ liệu mảng vector trả về giữ nguyên thứ tự sắp xếp của đầu vào
		const orderedData = (jsonResponse.data as Array<{
			object: string;
			embedding: number[] | string;
			index: number;
		}>).sort((a, b) => a.index - b.index);

		return orderedData.map((item) => {
			if (typeof item.embedding === 'string') {
				// Xử lý giải mã nếu người dùng kích hoạt truyền dữ liệu dạng Base64
				const binaryBuffer = Buffer.from(item.embedding, 'base64');
				if (this.outputDtype === 'float') {
					const floatArray = new Float32Array(
						binaryBuffer.buffer,
						binaryBuffer.byteOffset,
						binaryBuffer.byteLength / 4
					);
					return Array.from(floatArray);
				} else if (this.outputDtype === 'int8' || this.outputDtype === 'binary') {
					const intArray = new Int8Array(
						binaryBuffer.buffer,
						binaryBuffer.byteOffset,
						binaryBuffer.byteLength
					);
					return Array.from(intArray);
				} else {
					const uintArray = new Uint8Array(
						binaryBuffer.buffer,
						binaryBuffer.byteOffset,
						binaryBuffer.byteLength
					);
					return Array.from(uintArray);
				}
			}
			return item.embedding as number[];
		});
	}

	async embedDocuments(documents: string[]): Promise<number[][]> {
		return await this.generateEmbeddings(documents);
	}

	async embedQuery(document: string): Promise<number[]> {
		const result = await this.generateEmbeddings([document]);
		return result[0];
	}
}

let prototypePatched = false;

function patchVoyageEmbeddingsPrototype() {
	if (prototypePatched) return;
	try {
		const dep = ['@langchain', 'core', 'embeddings'].join('/');
		const LangChainEmbeddingsClass = requireN8nDependency(dep).Embeddings;
		if (LangChainEmbeddingsClass && LangChainEmbeddingsClass.prototype) {
			Object.setPrototypeOf(VoyageEmbeddingsModel.prototype, LangChainEmbeddingsClass.prototype);
		}
		prototypePatched = true;
	} catch (e) {}
}

// Định nghĩa cấu trúc n8n Node UI
export class EmbeddingsVoyage implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Embeddings Voyage AI',
		name: 'embeddingsVoyage',
		icon: 'file:voyage.svg',
		group: ['transform'],
		version: 1,
		description: 'Generate vector embeddings using Voyage AI API',
		defaults: {
			name: 'Embeddings Voyage AI',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Embeddings'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.voyageai.com/',
					},
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiEmbedding],
		outputNames: ['Embeddings'],
		credentials: [
			{
				name: 'voyageApi',
				required: true,
			},
		],
		properties: [
			getAiUtilities().getConnectionHintNoticeField([NodeConnectionTypes.AiVectorStore]),
			{
				displayName: 'Model',
				name: 'model',
				type: 'options',
				options: [
					{ name: 'Voyage-4 (Balanced performance & latency)', value: 'voyage-4' },
					{ name: 'Voyage-4-Large (Best quality multilingual)', value: 'voyage-4-large' },
					{ name: 'Voyage-4-Lite (Low latency & budget optimized)', value: 'voyage-4-lite' },
					{ name: 'Voyage-Code-3 (Optimized for code retrieval)', value: 'voyage-code-3' },
					{ name: 'Voyage-Finance-2 (Optimized for financial texts)', value: 'voyage-finance-2' },
					{ name: 'Voyage-Law-2 (Optimized for legal texts)', value: 'voyage-law-2' },
				],
				default: 'voyage-4',
				required: true,
				description: 'Select the embedding model appropriate for your data domain.',
			},
			{
				displayName: 'Input Type',
				name: 'inputType',
				type: 'options',
				options: [
					{ name: 'Document (Optimized for vector store storage)', value: 'document' },
					{ name: 'Not Specified (Default)', value: '' },
					{ name: 'Query (Optimized for search queries)', value: 'query' },
				],
				default: '',
				description: 'Specifies the type of input text for optimizing search relevance.',
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					{
						displayName: 'Output Dimension',
						name: 'outputDimension',
						type: 'options',
						options: [
							{ name: '256 dimensions', value: 256 },
							{ name: '512 dimensions', value: 512 },
							{ name: '1024 dimensions (Default)', value: 1024 },
							{ name: '2048 dimensions', value: 2048 },
						],
						default: 1024,
						description: 'Specify output vector dimensions using Matryoshka Representation Learning.',
					},
					{
						displayName: 'Output Data Type',
						name: 'outputDtype',
						type: 'options',
						options: [
							{ name: 'Binary (1-bit packed signed binary)', value: 'binary' },
							{ name: 'Float (Default 32-bit float)', value: 'float' },
							{ name: 'Int8 (8-bit signed integer)', value: 'int8' },
							{ name: 'Ubinary (1-bit packed unsigned binary)', value: 'ubinary' },
							{ name: 'Uint8 (8-bit unsigned integer)', value: 'uint8' },
						],
						default: 'float',
						description: 'Compress output vectors using quantization to save Vector DB storage.',
					},
					{
						displayName: 'Truncation',
						name: 'truncation',
						type: 'boolean',
						default: true,
						description: 'Whether to truncate the input texts if they exceed the context length limit.',
					},
					{
						displayName: 'Encoding Format',
						name: 'encodingFormat',
						type: 'options',
						options: [
							{ name: 'Base64 (Encoded binary string for bandwidth efficiency)', value: 'base64' },
							{ name: 'None (Standard float array)', value: '' },
						],
						default: '',
						description: 'Specifies the encoding format to return embeddings. Base64 reduces bandwidth and JSON parsing latency.',
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const modelName = this.getNodeParameter('model', itemIndex) as string;
		const inputType = this.getNodeParameter('inputType', itemIndex, '') as string;
		const options = this.getNodeParameter('options', itemIndex, {}) as {
			outputDimension?: number;
			outputDtype?: string;
			truncation?: boolean;
			encodingFormat?: string;
		};

		const credentials = await this.getCredentials('voyageApi');

		const baseUrl = (credentials.baseUrl as string) || 'https://api.voyageai.com/v1';
		const apiKey = credentials.apiKey as string;

		const embeddingProvider = new VoyageEmbeddingsModel({
			apiKey,
			baseUrl,
			modelName,
			inputType,
			outputDimension: options.outputDimension,
			outputDtype: options.outputDtype,
			truncation: options.truncation,
			encodingFormat: options.encodingFormat,
		}, this);

		return {
			response: getAiUtilities().logWrapper(embeddingProvider as any, this),
		};
	}
}
