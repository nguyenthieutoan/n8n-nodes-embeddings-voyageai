import {
	NodeConnectionTypes,
	NodeApiError,
	type INodeProperties,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type SupplyData,
} from 'n8n-workflow';
import { Embeddings, type EmbeddingsParams } from '@langchain/core/embeddings';
import { logWrapper } from '@n8n/ai-utilities';

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
export class VoyageEmbeddingsModel extends Embeddings {
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

// Thiết lập kế thừa prototype động để vượt qua kiểm tra instanceof trong logWrapper của @n8n/ai-utilities
try {
	const aiUtilitiesPath = require.resolve('@n8n/ai-utilities');
	const langchainEmbeddingsPath = require.resolve('@langchain/core/embeddings', { paths: [aiUtilitiesPath] });
	const ParentEmbeddingsClass = require(langchainEmbeddingsPath).Embeddings;
	if (ParentEmbeddingsClass && ParentEmbeddingsClass.prototype) {
		Object.setPrototypeOf(VoyageEmbeddingsModel.prototype, ParentEmbeddingsClass.prototype);
	}
} catch (e) {
	// Bỏ qua lỗi nếu không tìm thấy (ví dụ lúc build hoặc trong môi trường test cục bộ)
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
			response: logWrapper(embeddingProvider as any, this),
		};
	}
}
