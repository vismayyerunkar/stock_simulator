import { IGenericResponse } from "./genericModels"

export interface ILoginResponse extends IGenericResponse{
    data : {
        token: string
    }
}