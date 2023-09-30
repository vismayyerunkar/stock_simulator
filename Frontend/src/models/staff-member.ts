import { IGenericResponse } from 'src/models/genericModels';

export interface StaffMember {
  id: number;
  name: string;
  email: string;
}

export interface IStaffMemberResponse extends IGenericResponse {
  data: {
    count: number,
    staff: Array<StaffMember>
  };
}
