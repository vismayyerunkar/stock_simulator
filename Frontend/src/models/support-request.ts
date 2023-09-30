import { Category } from "./feedback";
import { IGenericResponse } from "./genericModels";

export interface SupportRequestUser {
    email: string,
    id: number,
    name: string,
    phone_number: string
}

export interface SupportRequest {
    id: number,
    category: Category,
    is_deleted: boolean,
    deleted_at: Date,
    created_on: Date,
    updated_on: Date,
    title: string,
    description: string,
    status: string,
    user: SupportRequestUser,
    assigned_to: number
}

export interface ISupportRequestListResponse extends IGenericResponse {
    data: {
        count: number,
        support_requests: Array<SupportRequest>
    }
}

export interface ISupportRequestDetailResponse extends IGenericResponse {
    data: SupportRequest
}